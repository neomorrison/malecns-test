"""Evaluate / film the trained locomotion controller through the hardware runtime.

    MUJOCO_GL=osmesa python scripts/eval_locomotion.py runs/loco/policy.npz --video out.mp4

Uses ``malecns_body.deploy`` (numpy only) exactly as a robot would.
Optional ``--sim2sim`` runs the body in a *differently configured* simulator
(1 ms timestep, RK4 integrator, altered friction / mass / servo gains) to test
that the controller does not rely on quirks of the training simulator.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import body as B  # noqa: E402
from malecns_body.deploy import LocomotionController, MujocoBody  # noqa: E402

# (duration s, vx, yaw, jump)
SCRIPT = [
    (2.0, 0.0, 0.0, False),
    (4.0, 1.2, 0.0, False),
    (3.0, 1.2, 0.6, False),
    (3.0, 1.5, -0.6, False),
    (4.0, 2.5, 0.0, False),
    (4.0, 3.5, 0.0, False),
    (1.5, 1.2, 0.0, False),
    (1.5, 1.0, 0.0, True),
    (2.0, 1.0, 0.0, False),
    (1.5, 0.0, 0.0, True),
    (2.0, 0.0, 0.0, False),
    (2.0, -0.4, 0.0, False),
    (2.0, 0.0, 0.8, False),
]


def make_body(sim2sim=False, seed=0):
    import mujoco
    m = mujoco.MjModel.from_xml_path(B.HUMANOID_XML)
    if sim2sim:
        rng = np.random.default_rng(seed)
        m.opt.timestep = 0.001
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        m.geom_friction[0, 0] = 0.7
        m.body_mass[:] *= rng.uniform(0.9, 1.1, m.nbody)
        m.actuator_gainprm[:, 0] *= 0.9
        m.actuator_biasprm[:, 1] *= 0.9
    return MujocoBody(m)


def run(policy, script=SCRIPT, video=None, sim2sim=False, seed=0, width=640, height=360, verbose=True):
    body = make_body(sim2sim, seed)
    ctl = LocomotionController(policy)
    ctl.set_joint_limits(body.lo, body.hi)
    body.reset()
    renderer = frames = None
    if video:
        import mujoco
        renderer = mujoco.Renderer(body.m, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance, cam.elevation, cam.azimuth = 4.5, -12, 120
        frames = []
    log = []
    fallen = False
    t = 0.0
    for dur, vx, yaw, jump in script:
        n = int(dur / B.CONTROL_DT)
        for k in range(n):
            quat, gyro = body.read_imu()
            q, dq = body.read_joints()
            target = ctl.step(quat, gyro, q, dq, vx, yaw, jump=(jump and k == 0))
            body.command_joints(target)
            body.advance()
            t += B.CONTROL_DT
            d = body.d
            yawang = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                                1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
            v = d.qvel[:2]
            vfwd = np.cos(yawang) * v[0] + np.sin(yawang) * v[1]
            log.append((t, vx, yaw, vfwd, d.qvel[5], d.qpos[2], ctl.jumping))
            if d.qpos[2] < 0.55:
                fallen = True
            if renderer is not None and k % 2 == 0:
                renderer.update_scene(d, cam)
                frames.append(renderer.render().copy())
        if fallen:
            break
    log = np.array(log, dtype=float)
    if verbose:
        print(f"{'FELL' if fallen else 'ok'} after {t:.1f}s")
        i = 0
        for dur, vx, yaw, jump in script:
            n = int(dur / B.CONTROL_DT)
            seg = log[i + n // 3:i + n]
            if len(seg) == 0:
                break
            print(f"  cmd vx={vx:+.1f} yaw={yaw:+.1f}{' JUMP' if jump else '    '} -> vx {seg[:,3].mean():+.2f}  "
                  f"yaw {seg[:,4].mean():+.2f}  pelvis z max {log[i:i+n,5].max():.2f}")
            i += n
    if video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=int(round(1 / (2 * B.CONTROL_DT))), quality=7, macro_block_size=8)
        print("wrote", video)
    return fallen, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--video", default=None)
    ap.add_argument("--sim2sim", action="store_true")
    args = ap.parse_args()
    run(args.policy, video=args.video, sim2sim=args.sim2sim)
