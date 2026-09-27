"""Vectorised, domain-randomised locomotion environment for the humanoid.

Design goal: a controller trained here should be deployable on a physical
body with the same kinematics. Therefore

* the policy (actor) only observes quantities a real robot measures on board:
  IMU gyro + gravity direction, joint encoders, its own previous action, and
  the command / gait clock. Ground-truth velocity, height, contact forces and
  physics parameters are given to the critic only (asymmetric actor-critic);
  the critic is discarded at deployment.
* every environment has its own physics model with randomised mass, centre of
  mass, friction, servo gains, motor strength, joint damping and armature,
  plus action latency, sensor noise and random pushes.
* joint torques and speeds are limited to the hardware actuator spec and the
  reward penalises power, torque, jerky actions and hard impacts.

Physics is stepped in C with ``mujoco.rollout`` across threads; all other
work is vectorised numpy, so there is no per-environment Python loop.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from .. import body as B


# ----------------------------------------------------------------------------
# small vectorised helpers
# ----------------------------------------------------------------------------

def quat_to_mat(q):
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices (body -> world)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_of(R):
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def to_heading(v, yaw):
    """Rotate world xy vectors into the heading (yaw-only) frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], axis=1)


def window(p, start, end, k=0.02):
    """Smooth indicator that phase p (in [0,1), circular) lies in [start, end)."""
    width = (end - start) % 1.0
    centre = (start + 0.5 * width) % 1.0
    d = np.abs(((p - centre) + 0.5) % 1.0 - 0.5)
    return 1.0 / (1.0 + np.exp(-(0.5 * width - d) / k))


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

@dataclass
class DRConfig:
    """Domain randomisation ranges (multiplicative unless noted)."""
    friction: tuple = (0.4, 1.3)          # absolute floor friction
    mass: tuple = (0.85, 1.15)            # per body
    torso_com: float = 0.03               # m, +- offset of torso COM
    kp: tuple = (0.8, 1.2)                # per actuator
    kd: tuple = (0.7, 1.3)
    strength: tuple = (0.8, 1.1)          # torque limit scale
    damping: tuple = (0.5, 1.5)
    armature: tuple = (0.7, 1.3)
    max_delay_substeps: int = 4           # action latency 0..16 ms
    push_interval: tuple = (3.0, 8.0)     # s
    push_vel: float = 0.6                 # m/s max xy kick
    push_ang: float = 0.4                 # rad/s max angular kick
    # uniform sensor noise amplitudes
    noise_gyro: float = 0.2
    noise_grav: float = 0.05
    noise_q: float = 0.01
    noise_qd: float = 1.5


@dataclass
class RewardConfig:
    lin_vel: float = 1.5
    yaw_rate: float = 1.0
    gait: float = 1.0
    clearance: float = -10.0
    slip: float = -0.3
    upright: float = -3.0
    torso: float = -2.0
    height: float = -30.0
    action_rate: float = -0.01
    torque: float = -1e-6
    power: float = -3e-5
    vel_limit: float = -0.3
    pos_limit: float = -3.0
    posture: float = -0.4
    feet_sep: float = -50.0
    feet_flat: float = -0.5
    impact: float = -2e-6
    alive: float = 0.3
    jump_height: float = 6.0
    jump_clear: float = 6.0
    termination: float = -10.0


@dataclass
class EnvConfig:
    episode_s: float = 20.0
    history: int = 5
    cmd_resample_s: tuple = (4.0, 8.0)
    vx_max: float = 1.5                   # raised by the curriculum up to 3.5
    vx_back: float = 0.5
    yaw_max: float = 1.0
    p_stand: float = 0.12
    p_back: float = 0.08
    jumps: bool = False                   # enabled by the curriculum
    jump_interval: tuple = (2.5, 6.0)
    pushes: bool = False                  # enabled by the curriculum
    p_arm_override: float = 0.25          # random arm postures (feeding etc.)
    randomize: bool = True
    dr: DRConfig = field(default_factory=DRConfig)
    rew: RewardConfig = field(default_factory=RewardConfig)


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

class LocomotionEnv:
    PROPRIO_DIM = 3 + 3 + B.N_ALL + B.N_ALL + B.N_LEG
    TASK_DIM = 9
    PRIV_DIM = 18

    def __init__(self, num_envs: int, cfg: EnvConfig | None = None, seed: int = 0, nthread: int = 4):
        self.cfg = cfg or EnvConfig()
        self.N = num_envs
        self.rng = np.random.default_rng(seed)
        self.base = mujoco.MjModel.from_xml_path(B.HUMANOID_XML)
        m = self.base
        assert abs(m.opt.timestep - B.SIM_DT) < 1e-9
        self.models = [copy.copy(m) for _ in range(num_envs)]
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.roller = rollout.Rollout(nthread=nthread)
        self.spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.nstate = mujoco.mj_stateSize(m, self.spec)
        self.nq, self.nv = m.nq, m.nv
        assert self.nstate == 1 + m.nq + m.nv + m.na, "unexpected state layout"

        name = lambda t, i: mujoco.mj_id2name(m, t, i)
        act_names = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        assert act_names == B.ALL_JOINTS, act_names
        jids = m.actuator_trnid[:, 0]
        self.qadr = m.jnt_qposadr[jids].copy()
        self.dadr = m.jnt_dofadr[jids].copy()
        lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.q_lo, self.q_hi = lo, hi
        self.q_soft_lo, self.q_soft_hi = mid - 0.92 * half, mid + 0.92 * half

        def sadr(n):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
            return int(m.sensor_adr[i])
        self.s = {n: sadr(n) for n in [
            "imu_gyro", "imu_quat", "sole_l", "sole_r", "foot_l_pos", "foot_r_pos",
            "foot_l_vel", "foot_r_vel", "torso_up", "foot_l_up", "foot_r_up",
            "head_pos", "com", "com_vel"]}
        self.s_tau = sadr("tau_lumbar_bend")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.torso_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.weight = float(m.body_subtreemass[1] * 9.81)

        # nominal parameters for randomisation
        self._nom = dict(
            mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), ipos=m.body_ipos.copy(),
            gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy(),
            frange=m.actuator_forcerange.copy(), damping=m.dof_damping.copy(),
            armature=m.dof_armature.copy())

        N = num_envs
        self.state = np.zeros((N, self.nstate))
        self.ctrl_prev = np.tile(B.DEFAULT_POSE, (N, 1))
        self.last_action = np.zeros((N, B.N_LEG))
        self.hist = np.zeros((N, self.cfg.history, self.PROPRIO_DIM))
        self.t = np.zeros(N)
        self.steps = np.zeros(N, dtype=np.int64)
        self.max_steps = int(self.cfg.episode_s / B.CONTROL_DT)
        self.phase = np.zeros(N)
        self.cmd = np.zeros((N, 2))
        self.cmd_timer = np.zeros(N)
        self.jump_t = np.full(N, -1.0)
        self.jump_timer = np.zeros(N)
        self.jump_peak = np.zeros(N)
        self.push_timer = np.zeros(N)
        self.delay = np.zeros(N, dtype=np.int64)
        self.arm_override = np.zeros(N, dtype=bool)
        self.arm_target = np.tile(B.DEFAULT_ARMS, (N, 1))
        self.arm_blend = np.zeros(N)
        self.dr_params = np.zeros((N, 5))  # friction, mass scale, kp scale, strength, delay
        self.wz_filt = np.zeros(N)           # heading rate, low-passed over ~0.3 s
        self.reset(np.arange(N))

    # ------------------------------------------------------------------ DR
    def _randomize(self, i):
        mdl, nom, dr, r = self.models[i], self._nom, self.cfg.dr, self.rng
        if not self.cfg.randomize:
            self.dr_params[i] = [mdl.geom_friction[self.floor, 0], 1.0, 1.0, 1.0, 0]
            return
        fr = r.uniform(*dr.friction)
        mdl.geom_friction[self.floor, 0] = fr
        ms = r.uniform(*dr.mass, size=mdl.nbody)
        ms[0] = 1.0
        mdl.body_mass[:] = nom["mass"] * ms
        mdl.body_inertia[:] = nom["inertia"] * ms[:, None]
        mdl.body_ipos[:] = nom["ipos"]
        mdl.body_ipos[self.torso_body] += r.uniform(-dr.torso_com, dr.torso_com, 3)
        kp = r.uniform(*dr.kp, size=mdl.nu)
        kd = r.uniform(*dr.kd, size=mdl.nu)
        mdl.actuator_gainprm[:, 0] = nom["gain"][:, 0] * kp
        mdl.actuator_biasprm[:, 1] = nom["bias"][:, 1] * kp
        mdl.actuator_biasprm[:, 2] = nom["bias"][:, 2] * kd
        st = r.uniform(*dr.strength, size=mdl.nu)
        mdl.actuator_forcerange[:] = nom["frange"] * st[:, None]
        mdl.dof_damping[:] = nom["damping"] * r.uniform(*dr.damping, size=mdl.nv)
        mdl.dof_armature[:] = nom["armature"] * r.uniform(*dr.armature, size=mdl.nv)
        self.delay[i] = r.integers(0, dr.max_delay_substeps + 1)
        total = (mdl.body_mass[1:]).sum() / nom["mass"][1:].sum()
        self.dr_params[i] = [fr, total, kp.mean(), st.mean(), self.delay[i] / max(1, dr.max_delay_substeps)]

    # --------------------------------------------------------------- reset
    def _sample_commands(self, idx):
        c, r, n = self.cfg, self.rng, len(idx)
        u = r.random(n)
        vx = r.uniform(0.2, c.vx_max, n)
        stand = u < c.p_stand
        back = (u >= c.p_stand) & (u < c.p_stand + c.p_back)
        vx[stand] = 0.0
        vx[back] = -r.uniform(0.1, c.vx_back, back.sum())
        yaw = r.uniform(-c.yaw_max, c.yaw_max, n)
        yaw[r.random(n) < 0.35] = 0.0
        # when standing, sometimes turn in place
        yaw[stand & (r.random(n) < 0.5)] = 0.0
        # running: gentler turns
        yaw = np.where(vx > 2.0, yaw * 0.5, yaw)
        self.cmd[idx, 0] = vx
        self.cmd[idx, 1] = yaw
        self.cmd_timer[idx] = r.uniform(*c.cmd_resample_s, n)
        # arm overrides (hands busy: feeding, carrying...)
        ov = r.random(n) < c.p_arm_override
        self.arm_override[idx] = ov
        lo, hi = self.q_lo[B.N_LEG:], self.q_hi[B.N_LEG:]
        tgt = r.uniform(lo, hi, (n, B.N_ARM))
        feed = r.random(n) < 0.5
        tgt[feed] = B.FEED_POSE + r.normal(0, 0.15, (feed.sum(), B.N_ARM))
        self.arm_target[idx] = np.clip(tgt, lo, hi)

    def reset(self, idx):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        r, n = self.rng, len(idx)
        for i in idx:
            self._randomize(i)
        self._sample_commands(idx)
        qpos = np.zeros((n, self.nq))
        qvel = np.zeros((n, self.nv))
        yaw = r.uniform(-np.pi, np.pi, n)
        qpos[:, 2] = B.DEFAULT_PELVIS_HEIGHT + 0.01
        qpos[:, 3] = np.cos(yaw / 2)
        qpos[:, 6] = np.sin(yaw / 2)
        qpos[:, self.qadr] = B.DEFAULT_POSE + r.normal(0, 0.04, (n, B.N_ALL))
        qpos[:, self.qadr] = np.clip(qpos[:, self.qadr], self.q_lo, self.q_hi)
        # start already moving at part of the commanded speed
        v0 = self.cmd[idx, 0] * r.uniform(0.0, 0.8, n)
        qvel[:, 0] = v0 * np.cos(yaw)
        qvel[:, 1] = v0 * np.sin(yaw)
        qvel[:, 6:] = r.normal(0, 0.05, (n, self.nv - 6))
        self.state[idx] = 0.0
        self.state[idx, 1:1 + self.nq] = qpos
        self.state[idx, 1 + self.nq:1 + self.nq + self.nv] = qvel
        self.ctrl_prev[idx] = B.DEFAULT_POSE
        self.last_action[idx] = 0.0
        self.t[idx] = 0.0
        self.steps[idx] = 0
        self.phase[idx] = r.random(n)
        self.jump_t[idx] = -1.0
        self.jump_timer[idx] = r.uniform(*self.cfg.jump_interval, n)
        self.jump_peak[idx] = 0.0
        self.push_timer[idx] = r.uniform(*self.cfg.dr.push_interval, n)
        self.arm_blend[idx] = 0.0
        self.wz_filt[idx] = 0.0
        self.hist[idx] = 0.0

    # ----------------------------------------------------------- stepping
    def _gait(self, idx=None):
        """Clock, per-foot desired swing indicator for envs ``idx`` (default all)."""
        idx = slice(None) if idx is None else idx
        vx, yaw = self.cmd[idx, 0], self.cmd[idx, 1]
        phase, jump_t = self.phase[idx], self.jump_t[idx]
        freq, swing, run_w = B.gait_params(vx)
        stand = (np.abs(vx) < 0.1) & (np.abs(yaw) < 0.1)
        in_jump = jump_t >= 0
        jf = np.clip(jump_t / B.JUMP_DURATION, 0.0, 1.0)
        p_l = phase % 1.0
        p_r = (phase + 0.5) % 1.0
        sw = np.stack([window(p_l, 0.0, swing), window(p_r, 0.0, swing)], 1)
        sw[stand] = 0.0
        jsw = window(jf, B.JUMP_TAKEOFF, B.JUMP_LANDING)
        sw[in_jump] = jsw[in_jump, None]
        return dict(freq=freq, swing=swing, run_w=run_w, stand=stand, in_jump=in_jump,
                    jump_frac=jf, sw=sw, p=np.stack([p_l, p_r], 1), phase=phase, vx=vx, yaw=yaw)

    def _arm_ctrl(self, g):
        arms = B.arm_swing_targets(g["phase"], g["vx"], g["run_w"], g["swing"])
        arms[g["stand"]] = B.DEFAULT_ARMS
        jarms = B.arm_jump_targets(g["jump_frac"])
        arms[g["in_jump"]] = jarms[g["in_jump"]]
        # blend towards an override posture (hands busy: feeding etc.)
        self.arm_blend += np.clip(self.arm_override.astype(float) - self.arm_blend, -0.05, 0.05)
        b = self.arm_blend[:, None]
        return (1 - b) * arms + b * self.arm_target

    @staticmethod
    def task_obs(g):
        return np.stack([
            g["vx"] / 2.0, g["yaw"],
            np.sin(2 * np.pi * g["phase"]), np.cos(2 * np.pi * g["phase"]),
            g["swing"], g["run_w"], g["stand"].astype(float),
            g["in_jump"].astype(float), g["jump_frac"] * g["in_jump"],
        ], axis=1)

    def _observe(self, SD, idx, fresh=False):
        """Actor (noisy, on-board sensors only) and critic observations for envs idx."""
        n = len(idx)
        st = self.state[idx]
        qpos = st[:, 1:1 + self.nq]
        qvel = st[:, 1 + self.nq:1 + self.nq + self.nv]
        q = qpos[:, self.qadr]
        qd = qvel[:, self.dadr]
        quat = SD[:, self.s["imu_quat"]:self.s["imu_quat"] + 4]
        R = quat_to_mat(quat)
        grav = -R[:, 2, :]
        gyro = SD[:, self.s["imu_gyro"]:self.s["imu_gyro"] + 3]
        g = self._gait(idx)
        la = self.last_action[idx]
        clean = np.concatenate([gyro * 0.25, grav, q - B.DEFAULT_POSE, qd * 0.05, la], 1)
        dr = self.cfg.dr
        if self.cfg.randomize:
            r = self.rng
            noisy = np.concatenate([
                (gyro + r.uniform(-1, 1, (n, 3)) * dr.noise_gyro) * 0.25,
                grav + r.uniform(-1, 1, (n, 3)) * dr.noise_grav,
                q - B.DEFAULT_POSE + r.uniform(-1, 1, (n, B.N_ALL)) * dr.noise_q,
                (qd + r.uniform(-1, 1, (n, B.N_ALL)) * dr.noise_qd) * 0.05,
                la], 1)
        else:
            noisy = clean
        if fresh:
            self.hist[idx] = noisy[:, None]
        else:
            self.hist[idx, 1:] = self.hist[idx, :-1]
            self.hist[idx, 0] = noisy
        task = self.task_obs(g)
        obs = np.concatenate([self.hist[idx].reshape(n, -1), task], 1)

        yaw = yaw_of(R)
        com_v = to_heading(SD[:, self.s["com_vel"]:self.s["com_vel"] + 3], yaw)
        fl = SD[:, self.s["foot_l_pos"]:self.s["foot_l_pos"] + 3]
        fr = SD[:, self.s["foot_r_pos"]:self.s["foot_r_pos"] + 3]
        fvl = to_heading(SD[:, self.s["foot_l_vel"]:self.s["foot_l_vel"] + 3], yaw)
        fvr = to_heading(SD[:, self.s["foot_r_vel"]:self.s["foot_r_vel"] + 3], yaw)
        F = SD[:, [self.s["sole_l"], self.s["sole_r"]]]
        wz = np.einsum("nj,nj->n", R[:, 2, :], qvel[:, 3:6])
        priv = np.concatenate([
            com_v, qvel[:, 2:3], qpos[:, 2:3] - B.DEFAULT_PELVIS_HEIGHT, self.wz_filt[idx, None] * 0.5,
            F / self.weight, fl[:, 2:3], fr[:, 2:3], fvl, fvr, self.dr_params[idx],
        ], 1)
        cobs = np.concatenate([clean, task, priv], 1)
        cache = dict(q=q, qd=qd, R=R, grav=grav, yaw=yaw, com_v=com_v, wz=wz, F=F,
                     fl=fl, fr=fr, fvl=fvl, fvr=fvr, g=g, qpos=qpos, qvel=qvel)
        return obs.astype(np.float32), cobs.astype(np.float32), cache

    def _forward_sensors(self, ids):
        d = self.datas[0]
        sd = np.zeros((len(ids), self.base.nsensordata))
        for k, i in enumerate(ids):
            mujoco.mj_setState(self.models[i], d, self.state[i], self.spec)
            mujoco.mj_forward(self.models[i], d)
            sd[k] = d.sensordata
        return sd

    def observe_all(self):
        ids = np.arange(self.N)
        obs, cobs, _ = self._observe(self._forward_sensors(ids), ids, fresh=True)
        return obs, cobs

    def step(self, action):
        cfg, N, r = self.cfg, self.N, self.rng
        action = np.clip(np.asarray(action, dtype=np.float64), -4.0, 4.0)
        g = self._gait()
        tg = B.leg_rhythm(g["phase"], g["vx"], g["yaw"], g["swing"], g["run_w"], g["stand"],
                          g["in_jump"], g["jump_frac"])
        legs = B.DEFAULT_POSE[:B.N_LEG] + tg + B.ACTION_SCALE * action
        ctrl = np.concatenate([legs, self._arm_ctrl(g)], 1)
        ctrl = np.clip(ctrl, self.q_lo, self.q_hi)
        ns = B.N_SUBSTEPS
        seq = np.repeat(ctrl[:, None], ns, axis=1)
        if cfg.randomize:
            k = np.arange(ns)[None, :] < self.delay[:, None]
            seq[k] = np.repeat(self.ctrl_prev[:, None], ns, axis=1)[k]

        if cfg.pushes:
            self.push_timer -= B.CONTROL_DT
            pk = self.push_timer <= 0
            if pk.any():
                o = 1 + self.nq
                self.state[pk, o:o + 2] += r.uniform(-1, 1, (pk.sum(), 2)) * cfg.dr.push_vel
                self.state[pk, o + 3:o + 6] += r.uniform(-1, 1, (pk.sum(), 3)) * cfg.dr.push_ang
                self.push_timer[pk] = r.uniform(*cfg.dr.push_interval, pk.sum())

        S, SD = self.roller.rollout(self.models, self.datas, self.state, seq)
        self.state = S[:, -1].copy()
        sd = SD[:, -1].copy()
        cols = [self.s["sole_l"], self.s["sole_r"]]
        sd[:, cols] = SD[:, :, cols].mean(1)
        self.ctrl_prev = ctrl
        prev_action = self.last_action
        self.last_action = action.copy()

        # advance clocks
        self.t += B.CONTROL_DT
        self.steps += 1
        self.phase = (self.phase + g["freq"] * B.CONTROL_DT) % 1.0
        in_jump = self.jump_t >= 0
        self.jump_t[in_jump] += B.CONTROL_DT
        self.jump_t[self.jump_t > B.JUMP_DURATION] = -1.0

        ids = np.arange(N)
        R_ = quat_to_mat(sd[:, self.s["imu_quat"]:self.s["imu_quat"] + 4])
        wz_now = np.einsum("nj,nj->n", R_[:, 2, :], self.state[:, 1 + self.nq + 3:1 + self.nq + 6])
        self.wz_filt += (wz_now - self.wz_filt) * (B.CONTROL_DT / 0.3)
        obs, cobs, cache = self._observe(sd, ids)
        rew, terms, fallen = self._reward(sd, cache, action, prev_action, g)
        timeout = self.steps >= self.max_steps
        bad = ~np.isfinite(self.state).all(1) | (np.abs(self.state).max(1) > 1e3)
        done = fallen | timeout | bad
        rew = np.where(bad, 0.0, rew)
        info = dict(terms=terms, timeout=timeout & ~fallen & ~bad, fallen=fallen | bad)

        # schedule next commands / jumps (take effect from the next step)
        if cfg.jumps:
            self.jump_timer -= B.CONTROL_DT
            start = (self.jump_timer <= 0) & (self.jump_t < 0)
            self.jump_t[start] = 0.0
            self.jump_timer[start] = r.uniform(*cfg.jump_interval, start.sum())
        self.cmd_timer -= B.CONTROL_DT
        resample = np.nonzero(self.cmd_timer <= 0)[0]
        if len(resample):
            self._sample_commands(resample)

        if done.any():
            dids = np.nonzero(done)[0]
            self.reset(dids)
            o2, c2, _ = self._observe(self._forward_sensors(dids), dids, fresh=True)
            obs[dids], cobs[dids] = o2, c2
        return obs, cobs, rew.astype(np.float32), done, info

    # ------------------------------------------------------------- reward
    def _reward(self, sd, c, action, prev_action, g):
        w = self.cfg.rew
        q, qd, R = c["q"], c["qd"], c["R"]
        qpos = c["qpos"]
        cmd_vx, cmd_yaw = self.cmd[:, 0], self.cmd[:, 1]
        terms = {}
        # --- task tracking
        sig = 0.25 + 0.1 * np.abs(cmd_vx)
        ev = (c["com_v"][:, 0] - cmd_vx) ** 2 + c["com_v"][:, 1] ** 2
        terms["lin_vel"] = w.lin_vel * np.exp(-ev / sig)
        # pelvis yaw oscillates with every step; track the low-passed heading rate
        terms["yaw_rate"] = w.yaw_rate * np.exp(-(self.wz_filt - cmd_yaw) ** 2 / 0.1)
        # --- gait pattern (periodic reward composition)
        F = c["F"]
        contact = np.clip(F / (0.08 * self.weight), 0.0, 1.0)
        sw = g["sw"]
        terms["gait"] = w.gait * (1.0 - np.abs(contact - (1.0 - sw))).mean(1)
        fz = np.stack([c["fl"][:, 2], c["fr"][:, 2]], 1)
        h_tgt = (0.09 + 0.05 * g["run_w"])[:, None]
        prof = np.sin(np.pi * np.clip(g["p"] / g["swing"][:, None], 0, 1))
        h_des = h_tgt * prof * (~g["in_jump"])[:, None]
        terms["clearance"] = w.clearance * (sw * np.maximum(0.0, h_des - fz)).sum(1)
        fv = np.stack([np.linalg.norm(c["fvl"], axis=1), np.linalg.norm(c["fvr"], axis=1)], 1)
        terms["slip"] = w.slip * (contact * (1 - sw) * fv ** 2).sum(1)
        # --- posture and stability
        terms["upright"] = w.upright * (c["grav"][:, :2] ** 2).sum(1)
        tu = sd[:, self.s["torso_up"]:self.s["torso_up"] + 3]
        terms["torso"] = w.torso * (tu[:, :2] ** 2).sum(1)
        z = qpos[:, 2]
        z_tgt = B.DEFAULT_PELVIS_HEIGHT - 0.02 - 0.02 * g["run_w"]
        herr = np.maximum(0.0, np.abs(z - z_tgt) - 0.04) ** 2
        terms["height"] = w.height * herr * (~g["in_jump"])
        li, ri = [B.ALL_JOINTS.index(j) for j in ("hip_abd_l", "hip_rot_l")], \
                 [B.ALL_JOINTS.index(j) for j in ("hip_abd_r", "hip_rot_r")]
        terms["posture"] = w.posture * ((q[:, li] ** 2).sum(1) + (q[:, ri] ** 2).sum(1)
                                        + (q[:, :3] ** 2).sum(1))
        rel = c["fl"][:, :2] - c["fr"][:, :2]
        sep = to_heading(rel, c["yaw"])[:, 1]
        terms["feet_sep"] = w.feet_sep * np.maximum(0.0, 0.14 - sep) ** 2
        fu = np.stack([sd[:, self.s["foot_l_up"]:self.s["foot_l_up"] + 2],
                       sd[:, self.s["foot_r_up"]:self.s["foot_r_up"] + 2]], 1)
        terms["feet_flat"] = w.feet_flat * (contact * (fu ** 2).sum(2)).sum(1)
        # --- effort and hardware limits
        tau = sd[:, self.s_tau:self.s_tau + B.N_ALL]
        terms["action_rate"] = w.action_rate * ((action - prev_action) ** 2).sum(1)
        terms["torque"] = w.torque * (tau[:, :B.N_LEG] ** 2).sum(1)
        terms["power"] = w.power * np.abs(tau * qd).sum(1)
        terms["vel_limit"] = w.vel_limit * np.maximum(0.0, np.abs(qd) - 0.9 * B.VEL_LIMIT).sum(1)
        terms["pos_limit"] = w.pos_limit * (np.maximum(0.0, q - self.q_soft_hi)
                                            + np.maximum(0.0, self.q_soft_lo - q)).sum(1)
        terms["impact"] = w.impact * (np.maximum(0.0, F - 1.6 * self.weight) ** 2).sum(1)
        # --- jumping
        in_air = g["in_jump"] & (g["jump_frac"] > B.JUMP_TAKEOFF) & (g["jump_frac"] < B.JUMP_LANDING)
        rise = np.clip(z - B.DEFAULT_PELVIS_HEIGHT, 0.0, 0.45)
        clear = np.clip(fz.min(1), 0.0, 0.4)
        terms["jump_height"] = w.jump_height * rise * in_air
        terms["jump_clear"] = w.jump_clear * clear * in_air
        # --- survival
        head_z = sd[:, self.s["head_pos"] + 2]
        fallen = (z < 0.55) | (tu[:, 2] < 0.4) | (head_z < 0.9)
        terms["alive"] = w.alive * np.ones(self.N)
        terms["termination"] = w.termination * fallen
        rew = sum(terms.values())
        return rew, terms, fallen

    # ------------------------------------------------------------ helpers
    @property
    def obs_dim(self):
        return self.cfg.history * self.PROPRIO_DIM + self.TASK_DIM

    @property
    def cobs_dim(self):
        return self.PROPRIO_DIM + self.TASK_DIM + self.PRIV_DIM

    @property
    def act_dim(self):
        return B.N_LEG
