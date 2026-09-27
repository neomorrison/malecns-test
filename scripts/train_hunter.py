"""Train the "regular model" hunter: a small MLP that maps the same senses the
fly brain gets to the same four commands, trained with PPO in the abstract
hunting world (malecns_body/hunting_env.py).

    python scripts/train_hunter.py --minutes 20 --out runs/hunter
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body.hunting_env import HuntingEnv  # noqa: E402
from malecns_body.locomotion import ppo  # noqa: E402
from malecns_body.locomotion.trainer import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--out", default="runs/hunter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = HuntingEnv(args.envs, seed=args.seed)

    def curriculum(it, window):
        if it % 100 == 0 and env.stats:
            s = env.stats[-500:]
            return f"fruit eaten per 60 s episode: {np.mean(s):.2f} (n={len(s)})"
        return None

    def summary(it, rec):
        s = env.stats[-200:]
        return f"fruit/episode={np.mean(s):.2f}" if s else ""

    train(env, args.out, args.minutes, ppo.PPOConfig(hidden=(64, 64), init_std=0.7, entropy_coef=0.002, horizon=32),
          seed=args.seed, curriculum=curriculum, summary=summary,
          export_meta=lambda it: dict(kind="hunter", n_features=env.obs_dim))


if __name__ == "__main__":
    main()
