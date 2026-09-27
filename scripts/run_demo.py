"""The maleCNS brain drives the humanoid body: hunt fruit, jump logs, eat.

    MUJOCO_GL=osmesa python scripts/run_demo.py --policy runs/loco/policy.npz --video results/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import connectome as C  # noqa: E402
from malecns_body.agent import FlyBrainedHumanoid  # noqa: E402
from malecns_body.circuits import DecoderConfig  # noqa: E402
from malecns_body.world import World, default_scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/locomotion_policy.npz")
    ap.add_argument("--video", default="results/demo.mp4")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--vmax", type=float, default=None, help="cap on decoded speed (m/s)")
    args = ap.parse_args()

    conn = C.load()
    sc = default_scenario(args.seed)
    if args.duration:
        sc.duration = args.duration
    world = World(sc, seed=args.seed)
    dec = DecoderConfig(v_max=args.vmax) if args.vmax else None
    agent = FlyBrainedHumanoid(conn, args.policy, world, decoder=dec, seed=args.seed)
    filmer = None
    frames = []
    if not args.no_video:
        from malecns_body.viz import Filmer
        filmer = Filmer(agent)
    n = int(sc.duration / 0.02)
    acc = np.zeros(conn.n, np.int64)
    trace = []
    t0 = time.time()
    for k in range(n):
        cmd = agent.step()
        acc += agent.last_counts
        trace.append(dict(t=round(agent.t, 3), behaviour=agent.behaviour, vx=round(agent.vx, 3),
                          yaw=round(agent.yaw, 3), hunger=round(agent.hunger, 3),
                          pos=[round(float(x), 3) for x in world.data.qpos[:3]],
                          rates={kk: [round(float(v), 1) for v in vv] for kk, vv in agent.decoder.rates.items()},
                          inputs={kk: round(float(v), 1) for kk, v in agent.inputs.items()}))
        if filmer is not None and k % 2 == 1:
            frames.append(filmer.frame(acc))
            acc[:] = 0
        if k % 250 == 0:
            print(f"t={agent.t:5.1f}s  {agent.behaviour:6s} v={agent.vx:+.2f} yaw={agent.yaw:+.2f} "
                  f"hunger={agent.hunger:.2f}  GF={agent.decoder.rates['GF'].max():.0f} "
                  f"MN9={agent.decoder.rates['FEED'].max():.0f} steer={cmd['steer']:+.0f} "
                  f"fwd={cmd['fwd_rate']:.0f}Hz  ({time.time()-t0:.0f}s wall)", flush=True)
        if agent.fallen and agent.t > 2:
            pass
    print("\nEvents:")
    for t, e in agent.events + world.events:
        print(f"  {t:6.2f}s  {e}")
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    base = os.path.splitext(args.video)[0]
    with open(base + "_trace.json", "w") as f:
        json.dump(dict(events=sorted(agent.events + world.events), trace=trace), f)
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=25, quality=7, macro_block_size=8)
        print("wrote", args.video)


if __name__ == "__main__":
    main()
