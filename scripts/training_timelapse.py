"""Film how the body learns: the same test course at successive training stages.

    MUJOCO_GL=osmesa python scripts/training_timelapse.py --snapshots runs/loco/snapshots \
        --log runs/loco/train_log.jsonl --out results/training_timelapse.mp4

Each clip runs the deployable controller from a policy snapshot through
stand -> walk -> turn -> jump, captioned with the training iteration and how
much simulated practice the body had at that point.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import jax
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import body as B  # noqa: E402
from malecns_body.deploy import LocomotionController, MujocoBody  # noqa: E402
from malecns_body.locomotion import ppo  # noqa: E402
from malecns_body.locomotion.env import LocomotionEnv  # noqa: E402
from malecns_body.viz import _font  # noqa: E402

COURSE = [(1.0, 0.0, 0.0, False), (3.0, 1.2, 0.0, False), (2.0, 1.2, 0.7, False),
          (1.5, 1.2, -0.7, False), (1.5, 0.8, 0.0, True)]


def untrained_policy(path):
    env_dims = LocomotionEnv.PROPRIO_DIM * 5 + LocomotionEnv.TASK_DIM
    params = ppo.init_params(jax.random.PRNGKey(0), env_dims, 10, B.N_LEG, ppo.PPOConfig())
    norm = ppo.RunningNorm(env_dims)
    meta = dict(joints_out=B.LEG_JOINTS, joints_in=B.ALL_JOINTS, default_pose=B.DEFAULT_POSE.tolist(),
                action_scale=B.ACTION_SCALE, control_dt=B.CONTROL_DT, history=5,
                proprio_dim=LocomotionEnv.PROPRIO_DIM, task_dim=LocomotionEnv.TASK_DIM, iteration=0)
    ppo.export_policy(path, params, norm, meta)


def film(policy, caption, sub, width=800, height=450):
    import mujoco
    body = MujocoBody()
    ctl = LocomotionController(policy)
    ctl.set_joint_limits(body.lo, body.hi)
    body.reset()
    r = mujoco.Renderer(body.m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance, cam.elevation, cam.azimuth = 4.2, -12, 115
    f_big, f = _font(22, True), _font(15)
    frames, t, fell_at = [], 0.0, None
    total = sum(c[0] for c in COURSE)
    for dur, vx, yaw, jump in COURSE:
        for k in range(int(dur / B.CONTROL_DT)):
            quat, gyro = body.read_imu()
            q, dq = body.read_joints()
            body.command_joints(ctl.step(quat, gyro, q, dq, vx, yaw, jump=(jump and k == 0)))
            body.advance()
            t += B.CONTROL_DT
            if body.d.qpos[2] < 0.5 and fell_at is None:
                fell_at = t
            if k % 2 == 0:
                r.update_scene(body.d, cam)
                im = Image.fromarray(r.render())
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, width, 62], fill=(18, 20, 26))
                d.text((14, 8), caption, font=f_big, fill=(235, 237, 242))
                d.text((14, 38), sub, font=f, fill=(150, 156, 170))
                cmd = "stand" if vx == 0 else (f"walk {vx:.1f} m/s" + (f", turn {'left' if yaw > 0 else 'right'}" if yaw else ""))
                if jump and k < 45:
                    cmd = "JUMP!"
                d.text((width - 210, 38), f"command: {cmd}", font=f, fill=(120, 200, 140))
                if fell_at is not None:
                    d.text((width - 210, 8), "fell over", font=f_big, fill=(235, 90, 90))
                d.rectangle([0, height - 5, int(width * t / total), height], fill=(90, 200, 120))
                frames.append(np.asarray(im))
    return frames, fell_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="runs/loco/snapshots")
    ap.add_argument("--log", default="runs/loco/train_log.jsonl")
    ap.add_argument("--out", default="results/training_timelapse.mp4")
    ap.add_argument("--max-clips", type=int, default=6)
    args = ap.parse_args()
    log = {}
    if os.path.exists(args.log):
        offset, prev = 0, 0
        for line in open(args.log):
            rec = json.loads(line)
            if rec["samples"] < prev:          # counter restarts when training is resumed
                offset += prev
            prev = rec["samples"]
            rec["samples"] += offset
            log[rec["it"]] = rec
    snaps = sorted(glob.glob(os.path.join(args.snapshots, "policy_it*.npz")),
                   key=lambda p: int(re.findall(r"it(\d+)", p)[0]))
    its = [int(re.findall(r"it(\d+)", p)[0]) for p in snaps]
    if len(snaps) > args.max_clips - 1:
        keep = np.unique(np.linspace(0, len(snaps) - 1, args.max_clips - 1).round().astype(int))
        snaps, its = [snaps[i] for i in keep], [its[i] for i in keep]
    u = os.path.join(args.snapshots, "untrained.npz")
    untrained_policy(u)
    snaps, its = [u] + snaps, [0] + its
    frames = []
    for p, it in zip(snaps, its):
        rec = log.get(it) or (log[max(k for k in log if k <= it)] if log and it > 0 else None)
        if it == 0:
            cap, sub = "Before training", "random neural network - it has never moved this body"
        else:
            hours = rec["samples"] * B.CONTROL_DT / 3600 if rec else 0
            cap = f"Training iteration {it:,}"
            sub = f"{rec['samples'] / 1e6:.1f} M practice steps  =  {hours:,.0f} hours of simulated experience" if rec else ""
        clip, fell = film(p, cap, sub)
        print(f"it={it}: {len(clip)} frames, {'fell at %.1fs' % fell if fell else 'stayed up'}", flush=True)
        frames += clip
        frames += [clip[-1]] * 10
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(args.out, frames, fps=25, quality=7, macro_block_size=8)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
