"""Hanging-fruit task: the body must discover getting up to reach high fruit.

    python scripts/train_reach.py --minutes 180 --out runs/reach

No reward mentions posture. A curriculum raises the fruit (not the body) once
the policy reliably reaches the current heights; above ~1.6 m the only way to
the fruit is to stand up and stretch.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body.locomotion import ppo  # noqa: E402
from malecns_body.locomotion.reach_env import REACH_ACTION_SCALE, FruitReachEnv, ReachConfig  # noqa: E402
from malecns_body.locomotion.trainer import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--out", default="runs/reach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = ReachConfig()
    env = FruitReachEnv(args.envs, cfg, seed=args.seed)
    print(f"fallen-pose bank: {len(env.bank)} states", flush=True)

    def curriculum(it, window):
        eaten = env.ep_eaten[-500:]
        env.ep_eaten = env.ep_eaten[-500:]
        success = float(np.mean(np.array(eaten) > 0)) if eaten else 0.0
        standing = float(np.mean([(i["pelvis_z"] > 0.8).mean() for i in window]))
        msg = None
        if success > 0.7 and len(eaten) >= 100 and cfg.fruit_h_max < cfg.fruit_h_final:
            cfg.fruit_h_max = round(min(cfg.fruit_h_final, cfg.fruit_h_max + 0.1), 2)
            env.ep_eaten = []
            msg = f"fruit raised: max height {cfg.fruit_h_max:.2f} m"
        if it % 100 == 0 or msg:
            return (msg or "") + f"  (episodes with fruit {success:.2f}, time standing {standing:.2f})"
        return msg

    def summary(it, rec):
        t = rec["terms"]
        return f"progress={t['progress']:.3f} fruit/step={t['fruit']:.3f} h_max={cfg.fruit_h_max:.2f}"

    def export_meta(it):
        return dict(kind="reach", action_scale=REACH_ACTION_SCALE, history=cfg.history,
                    proprio_dim=env.PROPRIO_DIM, task_dim=env.TASK_DIM, fruit_h_max=cfg.fruit_h_max)

    def restore(extra):
        cfg.fruit_h_max = extra.get("fruit_h_max", cfg.fruit_h_max)

    train(env, args.out, args.minutes, ppo.PPOConfig(), seed=args.seed, resume=args.resume,
          curriculum=curriculum, summary=summary, export_meta=export_meta, restore_extra=restore)


if __name__ == "__main__":
    main()
