"""Fly brain vs trained RL model vs hand-coded rules, on the same body and scenarios.

    MUJOCO_GL=osmesa python scripts/compare_controllers.py --policy models/locomotion_policy.npz \
        --hunter models/hunter_policy.npz --seeds 0 1 2 3 4 --video-seed 0

Every controller drives the same learned locomotion controller through the same
randomised scenarios (fruit placement, fleeing prey, log timing). Writes
results/comparison.{json,md,png} and a side-by-side video.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import body as B  # noqa: E402
from malecns_body.agent import Agent  # noqa: E402
from malecns_body.controllers import FlyBrainController, HandCodedController, LearnedController  # noqa: E402
from malecns_body.world import World, random_scenario  # noqa: E402

ORDER = ["fly", "learned", "hand"]
LABEL = {"fly": "maleCNS fly brain", "learned": "trained RL model", "hand": "hand-coded rules"}


def make_controller(kind, conn, hunter, seed):
    if kind == "fly":
        return FlyBrainController(conn, seed=seed)
    if kind == "learned":
        return LearnedController(hunter)
    return HandCodedController()


def run_one(kind, seed, args, conn, video_path=None):
    world = World(random_scenario(seed, duration=args.duration), seed=seed)
    ctrl = make_controller(kind, conn, args.hunter, seed)
    agent = Agent(world, args.policy, ctrl)
    writer = renderer = None
    if video_path:
        import imageio.v2 as imageio
        import mujoco
        from PIL import Image, ImageDraw
        from malecns_body.viz import _font
        renderer = mujoco.Renderer(world.model, 360, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = world.model.body("pelvis").id
        cam.distance, cam.elevation, cam.azimuth = 7.0, -18, 135
        writer = imageio.get_writer(video_path, fps=25, quality=7, macro_block_size=8)
        f_big, f = _font(18, True), _font(13)
    n = int(args.duration / B.CONTROL_DT)
    t0 = time.time()
    for k in range(n):
        agent.step()
        if writer is not None and k % 2 == 1:
            renderer.update_scene(world.data, cam)
            im = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 640, 50], fill=(18, 20, 26))
            d.text((10, 5), LABEL[kind], font=f_big, fill=(235, 237, 242))
            hits = sum(lg.hit for lg in world.logs)
            clr = sum(lg.passed and not lg.hit for lg in world.logs)
            d.text((10, 29), f"fruit eaten {len(agent.stats['eaten'])}   logs cleared {clr} / tripped {hits}"
                   f"   {agent.behaviour}", font=f, fill=(160, 166, 180))
            d.text((580, 8), f"{agent.t:4.1f}s", font=f, fill=(160, 166, 180))
            writer.append_data(np.asarray(im))
    wall = time.time() - t0
    if writer is not None:
        writer.close()
    logs_seen = [lg for lg in world.logs if lg.passed or lg.hit]
    return dict(controller=kind, seed=seed, fruit=len(agent.stats["eaten"]), eaten_at=agent.stats["eaten"],
                first_fruit_s=agent.stats["eaten"][0] if agent.stats["eaten"] else None,
                logs_cleared=int(sum(lg.passed and not lg.hit for lg in logs_seen)),
                logs_tripped=int(sum(lg.hit for lg in logs_seen)), jumps=agent.stats["jumps"],
                falls=agent.stats["falls"], distance_m=round(agent.stats["distance"], 1),
                decide_ms=1000 * ctrl.compute_s / n, wall_s=round(wall, 1),
                realtime_factor=round(wall / args.duration, 2))


def tile_videos(paths, out):
    import imageio.v2 as imageio
    readers = [imageio.get_reader(p) for p in paths]
    w = imageio.get_writer(out, fps=25, quality=7, macro_block_size=8)
    for frames in zip(*readers):
        w.append_data(np.concatenate(frames, axis=1))
    w.close()


def report(results, out_dir):
    kinds = [k for k in ORDER if any(r["controller"] == k for r in results)]
    agg = {}
    for k in kinds:
        rs = [r for r in results if r["controller"] == k]
        first = [r["first_fruit_s"] for r in rs if r["first_fruit_s"] is not None]
        logs = sum(r["logs_cleared"] + r["logs_tripped"] for r in rs)
        agg[k] = dict(runs=len(rs), fruit_mean=float(np.mean([r["fruit"] for r in rs])),
                      fruit_per_run=[r["fruit"] for r in rs],
                      first_fruit_s=float(np.mean(first)) if first else None,
                      logs_cleared=sum(r["logs_cleared"] for r in rs), logs_total=logs,
                      falls=sum(r["falls"] for r in rs), decide_ms=float(np.mean([r["decide_ms"] for r in rs])),
                      realtime_factor=float(np.mean([r["realtime_factor"] for r in rs])))
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(dict(aggregate=agg, runs=results), f, indent=1)
    lines = ["| decision maker | fruit eaten / 60 s | first fruit (s) | logs cleared | falls | decision cost (ms) | x real time |",
             "|---|---|---|---|---|---|---|"]
    for k in kinds:
        a = agg[k]
        ff = f"{a['first_fruit_s']:.1f}" if a["first_fruit_s"] else "-"
        lines.append(f"| {LABEL[k]} | {a['fruit_mean']:.1f} ({', '.join(map(str, a['fruit_per_run']))}) | {ff} | "
                     f"{a['logs_cleared']}/{a['logs_total']} | {a['falls']} | {a['decide_ms']:.2f} | {a['realtime_factor']:.1f} |")
    with open(os.path.join(out_dir, "comparison.md"), "w") as f:
        f.write("# Decision makers compared on the same body\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/locomotion_policy.npz")
    ap.add_argument("--hunter", default="models/hunter_policy.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--controllers", nargs="+", default=ORDER)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--video-seed", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conn = None
    if "fly" in args.controllers:
        from malecns_body import connectome as C
        conn = C.load()
    results = []
    vids = []
    for seed in args.seeds:
        for kind in args.controllers:
            vp = os.path.join(args.out, f"compare_{kind}_seed{seed}.mp4") if seed == args.video_seed else None
            r = run_one(kind, seed, args, conn, vp)
            if vp:
                vids.append(vp)
            results.append(r)
            print(json.dumps(r), flush=True)
    report(results, args.out)
    if vids:
        tile_videos(vids, os.path.join(args.out, f"comparison_seed{args.video_seed}.mp4"))
        print("wrote side-by-side video")


if __name__ == "__main__":
    main()
