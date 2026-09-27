"""Generic PPO training loop for the humanoid environments (locomotion, get-up)."""
from __future__ import annotations

import json
import os
import time
from typing import Callable

import jax
import numpy as np

from . import ppo
from .. import body as B


def train(env, out: str, minutes: float, pcfg: ppo.PPOConfig | None = None, seed: int = 0,
          resume: str | None = None, curriculum: Callable | None = None,
          export_meta: Callable | None = None, summary: Callable | None = None,
          restore_extra: Callable | None = None, save_every: int = 100):
    """Train with PPO until ``minutes`` of wall time have passed.

    curriculum(it, window) -> str|None   called every 25 iterations with the
        per-step info collected since the last call; may change env.cfg.
    export_meta(it) -> dict              metadata stored in the exported policy.
    summary(it, rec) -> str              one-line progress text.
    restore_extra(extra)                 re-apply saved curriculum state on resume.
    """
    os.makedirs(out, exist_ok=True)
    pcfg = pcfg or ppo.PPOConfig()
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params = ppo.init_params(k0, env.obs_dim, env.cobs_dim, env.act_dim, pcfg)
    opt, update = ppo.make_update(pcfg)
    opt_state = opt.init(params)
    obs_norm, cobs_norm = ppo.RunningNorm(env.obs_dim), ppo.RunningNorm(env.cobs_dim)
    lr, it = pcfg.lr, 0
    extra_state = {}
    if resume:
        ck = ppo.load_checkpoint(resume)
        params = ck["params"]
        opt_state = opt.init(params)
        obs_norm.load(ck["obs_norm"])
        cobs_norm.load(ck["cobs_norm"])
        it = ck["extra"]["iteration"]
        lr = ck["extra"].get("lr", lr)
        extra_state = ck["extra"]
        if restore_extra:
            restore_extra(extra_state)

    obs, cobs = env.observe_all()
    obs_norm.update(obs)
    cobs_norm.update(cobs)
    T, N = pcfg.horizon, env.N
    buf = {k: np.zeros((T, N, d), np.float32) for k, d in
           [("obs", env.obs_dim), ("cobs", env.cobs_dim), ("act", env.act_dim), ("mean", env.act_dim)]}
    for k in ["logp", "val", "rew", "done"]:
        buf[k] = np.zeros((T, N), np.float32)
    log_path = os.path.join(out, "train_log.jsonl")
    t_start = time.time()
    ep_ret, ep_len = np.zeros(N), np.zeros(N)
    recent_ret, recent_len = [], []
    window = []
    samples = 0

    def save():
        extra = dict(extra_state, iteration=it, lr=lr)
        ppo.save_checkpoint(os.path.join(out, "checkpoint.pkl"), params, opt_state, obs_norm, cobs_norm, extra)
        meta = dict(joints_in=B.ALL_JOINTS, default_pose=B.DEFAULT_POSE.tolist(), control_dt=B.CONTROL_DT,
                    iteration=it)
        if export_meta:
            meta.update(export_meta(it))
        ppo.export_policy(os.path.join(out, "policy.npz"), params, obs_norm, meta)

    while (time.time() - t_start) / 60 < minutes:
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
            window.append(info)
        samples += T * N

        last_val = np.asarray(ppo.value_fn(params, cobs_norm(cobs)))
        adv, ret = ppo.gae(buf["rew"], buf["val"], buf["done"], last_val, pcfg.gamma, pcfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in buf.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        log_std = np.tile(np.asarray(params["log_std"]), (T * N, 1))
        mb = T * N // pcfg.minibatches
        kls = []
        for _ in range(pcfg.epochs):
            perm = np.random.permutation(T * N)
            for j in range(pcfg.minibatches):
                ix = perm[j * mb:(j + 1) * mb]
                batch = {k: flat[k][ix] for k in ["obs", "cobs", "act", "logp", "val", "adv", "ret", "mean"]}
                batch["log_std"] = log_std[ix]
                params, opt_state, st = update(params, opt_state, batch, np.float32(lr))
                kl = float(st["kl"])
                kls.append(kl)
                if kl > pcfg.desired_kl * 2.0:
                    lr = max(1e-5, lr / 1.5)
                elif 0.0 < kl < pcfg.desired_kl / 2.0:
                    lr = min(1e-2, lr * 1.5)
        it += 1
        if curriculum and it % 25 == 0:
            msg = curriculum(it, window)
            if msg:
                print(f"[curriculum it={it}] {msg}", flush=True)
            window = []
        elif not curriculum:
            window = []
        rr, rl = recent_ret[-200:], recent_len[-200:]
        rec = dict(it=it, minutes=(time.time() - t_start) / 60, samples=samples,
                   fps=T * N / (time.time() - t_it), ret=float(np.mean(rr)) if rr else 0.0,
                   ep_len=float(np.mean(rl)) if rl else 0.0, lr=lr, kl=float(np.mean(kls)),
                   std=float(np.exp(np.asarray(params["log_std"])).mean()),
                   terms={k: round(v, 4) for k, v in term_sums.items()})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            line = summary(it, rec) if summary else ""
            print(f"it={it} t={rec['minutes']:.1f}m sps={rec['fps']:.0f} ret={rec['ret']:.1f} "
                  f"std={rec['std']:.2f} lr={lr:.1e} kl={rec['kl']:.4f} {line}", flush=True)
        if it % save_every == 0:
            save()
    save()
    print("done", flush=True)
