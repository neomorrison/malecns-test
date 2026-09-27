"""Teach the humanoid body to stand, walk, run, turn and jump.

    python scripts/train_locomotion.py --minutes 180 --out runs/loco

A single command-conditioned policy is trained with PPO. Commands are
(forward speed, yaw rate, jump trigger) - exactly the interface the maleCNS
brain drives at run time. A curriculum raises the top speed from brisk
walking to running, and switches on random pushes and jumping once the basic
gait is stable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import body as B  # noqa: E402
from malecns_body.locomotion.env import EnvConfig, LocomotionEnv  # noqa: E402
from malecns_body.locomotion import ppo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--out", default="runs/loco")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--vx-max-final", type=float, default=3.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ecfg = EnvConfig()
    pcfg = ppo.PPOConfig()
    env = LocomotionEnv(args.envs, ecfg, seed=args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    params = ppo.init_params(k0, env.obs_dim, env.cobs_dim, env.act_dim, pcfg)
    opt, update = ppo.make_update(pcfg)
    opt_state = opt.init(params)
    obs_norm, cobs_norm = ppo.RunningNorm(env.obs_dim), ppo.RunningNorm(env.cobs_dim)
    lr = pcfg.lr
    it0 = 0
    if args.resume:
        ck = ppo.load_checkpoint(args.resume)
        params = jax.tree_util.tree_map(lambda x: x, ck["params"])
        opt_state = opt.init(params)
        obs_norm.load(ck["obs_norm"])
        cobs_norm.load(ck["cobs_norm"])
        ex = ck["extra"]
        it0 = ex["iteration"]
        ecfg.vx_max, ecfg.jumps, ecfg.pushes = ex["vx_max"], ex["jumps"], ex["pushes"]
        lr = ex.get("lr", lr)

    obs, cobs = env.observe_all()
    obs_norm.update(obs)
    cobs_norm.update(cobs)
    T, N = pcfg.horizon, args.envs
    buf = {k: np.zeros((T, N, d), np.float32) for k, d in
           [("obs", env.obs_dim), ("cobs", env.cobs_dim), ("act", env.act_dim), ("mean", env.act_dim)]}
    for k in ["logp", "val", "rew", "done"]:
        buf[k] = np.zeros((T, N), np.float32)

    log_path = os.path.join(args.out, "train_log.jsonl")
    t_start = time.time()
    ep_ret = np.zeros(N)
    ep_len = np.zeros(N)
    recent_ret, recent_len = [], []
    cur = dict(track=[], falls=0, steps=0)
    it = it0
    samples = 0
    while (time.time() - t_start) / 60 < args.minutes:
        t_it = time.time()
        term_sums = {}
        for t in range(T):
            on, cn = obs_norm(obs), cobs_norm(cobs)
            key, ka = jax.random.split(key)
            a, logp, v, mean = ppo.policy_step(params, on, cn, ka)
            a, mean = np.asarray(a), np.asarray(mean)
            buf["obs"][t], buf["cobs"][t], buf["act"][t], buf["mean"][t] = on, cn, a, mean
            buf["logp"][t], buf["val"][t] = np.asarray(logp), np.asarray(v)
            obs, cobs, rew, done, info = env.step(a)
            # bootstrap through time-outs (not true terminations)
            rew = rew + pcfg.gamma * buf["val"][t] * info["timeout"]
            buf["rew"][t], buf["done"][t] = rew, done
            obs_norm.update(obs)
            cobs_norm.update(cobs)
            for k, v_ in info["terms"].items():
                term_sums[k] = term_sums.get(k, 0.0) + float(v_.mean()) / T
            ep_ret += rew
            ep_len += 1
            if done.any():
                recent_ret.extend(ep_ret[done].tolist())
                recent_len.extend(ep_len[done].tolist())
                ep_ret[done] = 0
                ep_len[done] = 0
            cur["track"].append(float((info["terms"]["lin_vel"] / ecfg.rew.lin_vel).mean()))
            cur["falls"] += int(info["fallen"].sum())
            cur["steps"] += N
        samples += T * N
        t_col = time.time() - t_it

        last_val = np.asarray(ppo.value_fn(params, cobs_norm(cobs)))
        adv, ret = ppo.gae(buf["rew"], buf["val"], buf["done"], last_val, pcfg.gamma, pcfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in buf.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        flat["log_std"] = np.tile(np.asarray(params["log_std"]), (T * N, 1))
        mb = T * N // pcfg.minibatches
        kls = []
        for ep in range(pcfg.epochs):
            perm = np.random.permutation(T * N)
            for j in range(pcfg.minibatches):
                ix = perm[j * mb:(j + 1) * mb]
                batch = {k: flat[k][ix] for k in ["obs", "cobs", "act", "logp", "val", "adv", "ret", "mean"]}
                batch["log_std"] = flat["log_std"][ix]
                params, opt_state, st = update(params, opt_state, batch, np.float32(lr))
                kl = float(st["kl"])
                kls.append(kl)
                if kl > pcfg.desired_kl * 2.0:
                    lr = max(1e-5, lr / 1.5)
                elif 0.0 < kl < pcfg.desired_kl / 2.0:
                    lr = min(1e-2, lr * 1.5)
        it += 1

        # ---------------- curriculum
        if it % 25 == 0 and cur["steps"] > 0:
            track = float(np.mean(cur["track"]))
            fall_rate = cur["falls"] / cur["steps"] / B.CONTROL_DT  # falls per env-second
            msg = None
            if track > 0.62 and fall_rate < 0.05:
                if not ecfg.pushes:
                    ecfg.pushes = ecfg.jumps = True
                    msg = "pushes + jumps on"
                elif ecfg.vx_max < args.vx_max_final:
                    ecfg.vx_max = min(args.vx_max_final, ecfg.vx_max + 0.25)
                    msg = f"vx_max -> {ecfg.vx_max:.2f}"
            if msg:
                print(f"[curriculum it={it}] track={track:.2f} falls/s={fall_rate:.4f}: {msg}", flush=True)
            cur = dict(track=[], falls=0, steps=0)

        rr = recent_ret[-200:]
        rl = recent_len[-200:]
        rec = dict(it=it, minutes=(time.time() - t_start) / 60, samples=samples,
                   fps=T * N / (time.time() - t_it), collect_s=t_col,
                   ret=float(np.mean(rr)) if rr else 0.0, ep_len=float(np.mean(rl)) if rl else 0.0,
                   lr=lr, kl=float(np.mean(kls)), std=float(np.exp(np.asarray(params["log_std"])).mean()),
                   vx_max=ecfg.vx_max, pushes=ecfg.pushes, jumps=ecfg.jumps,
                   terms={k: round(v, 4) for k, v in term_sums.items()})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            print(f"it={it} t={rec['minutes']:.1f}m sps={rec['fps']:.0f} ret={rec['ret']:.1f} len={rec['ep_len']:.0f} "
                  f"trk={term_sums['lin_vel']/ecfg.rew.lin_vel:.2f} gait={term_sums['gait']:.2f} "
                  f"std={rec['std']:.2f} lr={lr:.1e} kl={rec['kl']:.4f} vmax={ecfg.vx_max:.2f} "
                  f"jump={term_sums['jump_height']:.3f}", flush=True)
        if it % 100 == 0:
            save(args.out, params, opt_state, obs_norm, cobs_norm, it, ecfg, lr, env)
    save(args.out, params, opt_state, obs_norm, cobs_norm, it, ecfg, lr, env)
    print("done", flush=True)


def save(out, params, opt_state, obs_norm, cobs_norm, it, ecfg, lr, env):
    extra = dict(iteration=it, vx_max=ecfg.vx_max, jumps=ecfg.jumps, pushes=ecfg.pushes, lr=lr)
    ppo.save_checkpoint(os.path.join(out, "checkpoint.pkl"), params, opt_state, obs_norm, cobs_norm, extra)
    meta = dict(
        joints_out=B.LEG_JOINTS, joints_in=B.ALL_JOINTS, default_pose=B.DEFAULT_POSE.tolist(),
        action_scale=B.ACTION_SCALE, control_dt=B.CONTROL_DT, history=env.cfg.history,
        proprio_dim=env.PROPRIO_DIM, task_dim=env.TASK_DIM, iteration=it, vx_max=ecfg.vx_max,
        jumps=ecfg.jumps)
    ppo.export_policy(os.path.join(out, "policy.npz"), params, obs_norm, meta)


if __name__ == "__main__":
    main()
