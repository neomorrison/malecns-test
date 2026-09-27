"""Hardware-facing runtime for the learned locomotion controller.

This module has **no dependency on JAX or MuJoCo**: a robot computer needs
only numpy and the exported ``policy.npz``. It reproduces, step for step,
what the policy saw in training:

    every 20 ms (50 Hz):
        read IMU (orientation quaternion, gyro) and joint encoders (q, dq)
        build the observation (5-frame history + command + gait clock)
        run the MLP -> leg joint offsets
        add the rhythm generator + default pose -> joint position targets
        add the arm motor program (swing / jump / feed)
        send targets to the joint servos (PD at >= 250 Hz, gains in body.ACTUATOR_SPEC)

The command interface is (forward speed m/s, yaw rate rad/s, jump trigger,
feed level) - the same four numbers the maleCNS brain produces - so any
body that implements ``RobotInterface`` can be driven by the fly brain.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np

from . import body as B


class RobotInterface(ABC):
    """What a physical (or simulated) body must provide."""

    @abstractmethod
    def read_imu(self):
        """-> (quat_wxyz (4,), gyro_rad_s (3,)) of the pelvis IMU, body frame."""

    @abstractmethod
    def read_joints(self):
        """-> (q (23,), dq (23,)) in body.ALL_JOINTS order, radians / rad/s."""

    @abstractmethod
    def command_joints(self, q_target):
        """Send 23 joint position targets (rad) to the PD servos."""


class LocomotionController:
    """Numpy re-implementation of the trained policy + rhythm + arm programs."""

    def __init__(self, policy_path: str):
        z = np.load(policy_path)
        self.nl = int(z["n_layers"])
        self.W = [z[f"W{i}"].astype(np.float64) for i in range(self.nl)]
        self.b = [z[f"b{i}"].astype(np.float64) for i in range(self.nl)]
        self.mean, self.std, self.clip = z["obs_mean"], z["obs_std"], float(z["obs_clip"])
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        assert self.meta["joints_in"] == B.ALL_JOINTS
        self.H = self.meta["history"]
        self.lo = np.array([-np.inf] * B.N_ALL)
        self.hi = np.array([np.inf] * B.N_ALL)
        self.reset()

    def set_joint_limits(self, lo, hi):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)

    def reset(self, phase: float = 0.0):
        self.hist = None
        self.last_action = np.zeros(B.N_LEG)
        self.phase = phase
        self.jump_t = -1.0
        self.feed = 0.0
        self.vx = 0.0
        self.yaw = 0.0

    # -- the network ---------------------------------------------------------
    def _mlp(self, x):
        for i in range(self.nl - 1):
            x = x @ self.W[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return x @ self.W[-1] + self.b[-1]

    def _gait(self):
        vx, yaw = np.array([self.vx]), np.array([self.yaw])
        freq, swing, run_w = B.gait_params(vx)
        stand = (np.abs(vx) < 0.1) & (np.abs(yaw) < 0.1)
        in_jump = np.array([self.jump_t >= 0])
        jf = np.clip(np.array([self.jump_t]) / B.JUMP_DURATION, 0, 1)
        return dict(freq=freq, swing=swing, run_w=run_w, stand=stand, in_jump=in_jump,
                    jump_frac=jf, phase=np.array([self.phase]), vx=vx, yaw=yaw)

    # -- one control tick ----------------------------------------------------
    def step(self, quat, gyro, q, dq, vx, yaw_rate, jump=False, feed=0.0):
        """Returns 23 joint position targets. Call every CONTROL_DT seconds."""
        self.vx, self.yaw, self.feed = float(vx), float(yaw_rate), float(feed)
        if jump and self.jump_t < 0:
            self.jump_t = 0.0
        w, x, y, zq = quat
        grav = -np.array([2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)])
        frame = np.concatenate([np.asarray(gyro) * 0.25, grav, np.asarray(q) - B.DEFAULT_POSE,
                                np.asarray(dq) * 0.05, self.last_action])
        if self.hist is None:
            self.hist = np.tile(frame, (self.H, 1))
        else:
            self.hist = np.roll(self.hist, 1, axis=0)
            self.hist[0] = frame
        g = self._gait()
        task = np.array([g["vx"][0] / 2.0, g["yaw"][0], np.sin(2 * np.pi * self.phase),
                         np.cos(2 * np.pi * self.phase), g["swing"][0], g["run_w"][0],
                         float(g["stand"][0]), float(g["in_jump"][0]),
                         g["jump_frac"][0] * float(g["in_jump"][0])])
        obs = np.concatenate([self.hist.reshape(-1), task])
        obs = np.clip((obs - self.mean) / self.std, -self.clip, self.clip)
        action = np.clip(self._mlp(obs), -4.0, 4.0)
        tg = B.leg_rhythm(g["phase"], g["vx"], g["yaw"], g["swing"], g["run_w"], g["stand"],
                          g["in_jump"], g["jump_frac"])[0]
        legs = B.DEFAULT_POSE[:B.N_LEG] + tg + B.ACTION_SCALE * action
        arms = self._arms(g)
        target = np.clip(np.concatenate([legs, arms]), self.lo, self.hi)
        # advance clocks exactly as in training
        self.last_action = action
        self.phase = (self.phase + g["freq"][0] * B.CONTROL_DT) % 1.0
        if self.jump_t >= 0:
            self.jump_t += B.CONTROL_DT
            if self.jump_t > B.JUMP_DURATION:
                self.jump_t = -1.0
        return target

    def _arms(self, g):
        arms = B.arm_swing_targets(g["phase"], g["vx"], g["run_w"], g["swing"])[0]
        if g["stand"][0]:
            arms = B.DEFAULT_ARMS.copy()
        if g["in_jump"][0]:
            arms = B.arm_jump_targets(g["jump_frac"])[0]
        if self.feed > 0:
            arms = (1 - self.feed) * arms + self.feed * B.FEED_POSE
        return arms

    @property
    def jumping(self):
        return self.jump_t >= 0


class MujocoBody(RobotInterface):
    """Simulated body implementing the hardware interface (for sim-to-sim tests)."""

    def __init__(self, model=None, data=None):
        import mujoco
        self.mj = mujoco
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(B.HUMANOID_XML)
        self.d = data if data is not None else mujoco.MjData(self.m)
        jids = self.m.actuator_trnid[:, 0]
        self.qadr = self.m.jnt_qposadr[jids]
        self.dadr = self.m.jnt_dofadr[jids]
        self.lo, self.hi = self.m.jnt_range[jids, 0], self.m.jnt_range[jids, 1]
        self.s_quat = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]
        self.s_gyro = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        self.substeps = int(round(B.CONTROL_DT / self.m.opt.timestep))

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.mj.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [x, y, B.DEFAULT_PELVIS_HEIGHT + 0.01]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qpos[self.qadr] = B.DEFAULT_POSE
        self.d.ctrl[:] = B.DEFAULT_POSE
        self.mj.mj_forward(self.m, self.d)

    def read_imu(self):
        return (self.d.sensordata[self.s_quat:self.s_quat + 4].copy(),
                self.d.sensordata[self.s_gyro:self.s_gyro + 3].copy())

    def read_joints(self):
        return self.d.qpos[self.qadr].copy(), self.d.qvel[self.dadr].copy()

    def command_joints(self, q_target):
        self.d.ctrl[:] = q_target

    def advance(self, callback=None):
        for _ in range(self.substeps):
            self.mj.mj_step(self.m, self.d)
            if callback is not None:
                callback()
