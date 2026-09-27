"""Pick the hanging fruit: a goal from which getting up has to be discovered.

Nothing in this environment describes standing. The body is given a goal -
touch a fruit hanging in the air with either hand - and pays the physical costs
a real body pays (energy, torque, jerky motion, joint limits). Episodes mostly
begin with the body lying on the ground (on its back, front or side, taken from
a bank of poses produced by letting it collapse from random configurations).

A curriculum raises the fruit, not the posture: low fruit can be reached lying
or sitting, fruit at ~1.3-1.5 m needs kneeling, and fruit above ~1.6 m can only
be reached standing with an arm stretched up (max reach 1.94 m). If the policy
stands up, it is because standing is how you get the fruit.

Reward = progress of the nearest hand towards the fruit (potential-based
shaping, Ng et al. 1999: it leaves the optimal behaviour unchanged) + a bonus
per fruit reached, after which a new fruit appears nearby.

Sim-to-real conditions match the locomotion environment: on-board sensors
(IMU, encoders, action history) plus the fruit position as seen by the body
(noisy, in the body frame); per-body domain randomisation, latency, noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from .. import body as B
from .env import DRConfig, HumanoidSim, quat_to_mat

REACH_ACTION_SCALE = 0.6


@dataclass
class ReachRewardConfig:
    progress: float = 10.0        # per metre the nearest hand gets closer
    fruit: float = 5.0            # per fruit reached
    action_rate: float = -0.005
    torque: float = -1e-6
    power: float = -2e-5
    vel_limit: float = -0.3
    pos_limit: float = -1.0
    spin: float = -0.01


@dataclass
class ReachConfig:
    episode_s: float = 8.0
    history: int = 5
    bank_size: int = 2048
    p_crouch: float = 0.15        # start crouched instead of lying
    p_standing: float = 0.15      # start standing
    fruit_dist: tuple = (0.3, 1.0)    # horizontal distance of a new fruit from the pelvis (m)
    fruit_h_max: float = 1.0      # raised by the curriculum up to fruit_h_final
    fruit_h_final: float = 1.85
    fruit_h_span: float = 0.6     # heights are drawn from [h_max - span, h_max]
    reach_radius: float = 0.12
    vision_noise: float = 0.03    # m
    randomize: bool = True
    dr: DRConfig = field(default_factory=DRConfig)
    rew: ReachRewardConfig = field(default_factory=ReachRewardConfig)


def _quat_from_euler(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)


def make_fallen_bank(model, n, rng, settle_s=2.0, nthread=4):
    """Drop the body from random orientations / limb postures and let it settle."""
    spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ns = mujoco.mj_stateSize(model, spec)
    jids = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[jids]
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    out = []
    datas = [mujoco.MjData(model) for _ in range(nthread)]
    steps = int(settle_s / model.opt.timestep)
    while sum(len(o) for o in out) < n:
        k = 512
        kind = rng.integers(0, 3, k)                      # 0 back, 1 front, 2 side
        roll = np.where(kind == 2, rng.choice([-1, 1], k) * np.pi / 2, 0) + rng.normal(0, 0.3, k)
        pitch = np.where(kind == 0, -np.pi / 2, np.where(kind == 1, np.pi / 2, 0)) + rng.normal(0, 0.3, k)
        yaw = rng.uniform(-np.pi, np.pi, k)
        st = np.zeros((k, ns))
        qpos = np.zeros((k, model.nq))
        qpos[:, 2] = 0.35
        qpos[:, 3:7] = _quat_from_euler(roll, pitch, yaw)
        pose = rng.uniform(lo, hi, (k, model.nu)) * 0.6 + B.DEFAULT_POSE * 0.4
        qpos[:, qadr] = pose
        st[:, 1:1 + model.nq] = qpos
        S, _ = rollout.rollout([model] * k, datas, st, np.repeat(pose[:, None], steps, 1))
        fin = S[:, -1]
        z = fin[:, 3]
        qv = np.abs(fin[:, 1 + model.nq:]).max(1)
        ok = (z < 0.45) & (qv < 1.0) & np.isfinite(fin).all(1)
        out.append(fin[ok])
    return np.concatenate(out)[:n]


class FruitReachEnv(HumanoidSim):
    PROPRIO_DIM = 3 + 3 + B.N_ALL + B.N_ALL + B.N_ALL
    TASK_DIM = 4
    PRIV_DIM = 16

    def __init__(self, num_envs: int, cfg: ReachConfig | None = None, seed: int = 0, nthread: int = 4):
        self.cfg = cfg or ReachConfig()
        super().__init__(num_envs, self.cfg.dr, self.cfg.randomize, seed, nthread)
        N = num_envs
        self.bank = make_fallen_bank(self.base, self.cfg.bank_size, self.rng, nthread=nthread)
        m = self.base
        self.hand_sites = [m.site("hand_l").id, m.site("hand_r").id]
        self.hand_bodies = [m.site_bodyid[s] for s in self.hand_sites]
        self.hand_offsets = [m.site_pos[s].copy() for s in self.hand_sites]
        self.state = np.zeros((N, self.nstate))
        self.ctrl_prev = np.tile(B.DEFAULT_POSE, (N, 1))
        self.last_action = np.zeros((N, B.N_ALL))
        self.hist = np.zeros((N, self.cfg.history, self.PROPRIO_DIM))
        self.steps = np.zeros(N, dtype=np.int64)
        self.max_steps = int(self.cfg.episode_s / B.CONTROL_DT)
        self.fruit = np.zeros((N, 3))
        self.prev_dist = np.zeros(N)
        self.eaten = np.zeros(N, dtype=np.int64)
        self.ep_eaten = []             # fruits per finished episode (for the curriculum)
        self.hands = np.zeros((N, 2, 3))
        self.reset(np.arange(N))

    # ------------------------------------------------------------------ fruit
    def _spawn_fruit(self, idx, pelvis_xy):
        c, r, n = self.cfg, self.rng, len(idx)
        d = r.uniform(*c.fruit_dist, n)
        a = r.uniform(-np.pi, np.pi, n)
        h = r.uniform(max(0.25, c.fruit_h_max - c.fruit_h_span), c.fruit_h_max, n)
        self.fruit[idx, 0] = pelvis_xy[:, 0] + d * np.cos(a)
        self.fruit[idx, 1] = pelvis_xy[:, 1] + d * np.sin(a)
        self.fruit[idx, 2] = h

    def _hands_from_state(self, ids):
        """Hand positions via forward kinematics (per env, only when needed)."""
        d = self.datas[0]
        out = np.zeros((len(ids), 2, 3))
        for k, i in enumerate(ids):
            mujoco.mj_setState(self.models[i], d, self.state[i], self.spec)
            mujoco.mj_kinematics(self.models[i], d)
            out[k] = d.site_xpos[self.hand_sites]
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, idx):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        r, n = self.rng, len(idx)
        for i in idx:
            self._randomize(i)
        st = self.bank[r.integers(0, len(self.bank), n)].copy()
        st[:, 0] = 0.0
        qpos = st[:, 1:1 + self.nq]
        qpos[:, :2] = 0.0
        yaw = r.uniform(-np.pi, np.pi, n)
        qz = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], 1)
        for k in range(n):
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qz[k], qpos[k, 3:7])
            qpos[k, 3:7] = q
        u = r.random(n)
        crouch = u < self.cfg.p_crouch
        standing = (u >= self.cfg.p_crouch) & (u < self.cfg.p_crouch + self.cfg.p_standing)
        for k in np.nonzero(crouch | standing)[0]:
            qpos[k] = 0.0
            qpos[k, 3] = np.cos(yaw[k] / 2)
            qpos[k, 6] = np.sin(yaw[k] / 2)
            pose = B.DEFAULT_POSE.copy()
            if crouch[k]:
                depth = r.uniform(0.4, 1.0)
                for side in ("l", "r"):
                    pose[B.ALL_JOINTS.index("hip_flex_" + side)] += 1.4 * depth
                    pose[B.ALL_JOINTS.index("knee_" + side)] += 2.0 * depth
                    pose[B.ALL_JOINTS.index("ankle_flex_" + side)] += 0.35 * depth
                pose[B.ALL_JOINTS.index("lumbar_flex")] += 0.4 * depth
                pose += r.normal(0, 0.1, B.N_ALL)
            qpos[k, self.qadr] = np.clip(pose, self.q_lo, self.q_hi)
            qpos[k, 2] = self._feet_on_floor_height(qpos[k])
        st[:, 1 + self.nq:] = 0.0
        self.state[idx] = st
        self.ctrl_prev[idx] = qpos[:, self.qadr]
        self.last_action[idx] = 0.0
        self.steps[idx] = 0
        self.eaten[idx] = 0
        self.hist[idx] = 0.0
        self._spawn_fruit(idx, np.zeros((n, 2)))
        self.hands[idx] = self._hands_from_state(idx)
        self.prev_dist[idx] = self._dist(idx)

    def _feet_on_floor_height(self, qpos):
        d = self.datas[0]
        d.qpos[:] = qpos
        d.qpos[2] = 1.5
        mujoco.mj_kinematics(self.base, d)
        sole = min(d.site("foot_l").xpos[2], d.site("foot_r").xpos[2])
        return 1.5 - sole + 0.01

    def _dist(self, idx):
        return np.linalg.norm(self.hands[idx] - self.fruit[idx, None], axis=2).min(1)

    # --------------------------------------------------------------- observe
    def _observe(self, SD, idx, fresh=False):
        n = len(idx)
        st = self.state[idx]
        qpos = st[:, 1:1 + self.nq]
        qvel = st[:, 1 + self.nq:1 + self.nq + self.nv]
        q, qd = qpos[:, self.qadr], qvel[:, self.dadr]
        R = quat_to_mat(SD[:, self.s["imu_quat"]:self.s["imu_quat"] + 4])
        grav = -R[:, 2, :]
        gyro = SD[:, self.s["imu_gyro"]:self.s["imu_gyro"] + 3]
        la = self.last_action[idx]
        clean = np.concatenate([gyro * 0.25, grav, q - B.DEFAULT_POSE, qd * 0.05, la], 1)
        dr = self.cfg.dr
        if self.cfg.randomize:
            r = self.rng
            noisy = np.concatenate([
                (gyro + r.uniform(-1, 1, (n, 3)) * dr.noise_gyro) * 0.25,
                grav + r.uniform(-1, 1, (n, 3)) * dr.noise_grav,
                q - B.DEFAULT_POSE + r.uniform(-1, 1, (n, B.N_ALL)) * dr.noise_q,
                (qd + r.uniform(-1, 1, (n, B.N_ALL)) * dr.noise_qd) * 0.05, la], 1)
        else:
            noisy = clean
        if fresh:
            self.hist[idx] = noisy[:, None]
        else:
            self.hist[idx, 1:] = self.hist[idx, :-1]
            self.hist[idx, 0] = noisy
        # the fruit as the body sees it: vector from pelvis, in the pelvis frame
        rel_w = self.fruit[idx] - qpos[:, :3]
        rel_b = np.einsum("nji,nj->ni", R, rel_w)
        seen = rel_b + (self.rng.normal(0, self.cfg.vision_noise, rel_b.shape) if self.cfg.randomize else 0)
        task = np.concatenate([seen * 0.5, np.linalg.norm(seen, axis=1, keepdims=True) * 0.5], 1)
        obs = np.concatenate([self.hist[idx].reshape(n, -1), task], 1)
        tu = SD[:, self.s["torso_up"]:self.s["torso_up"] + 3]
        com_v = SD[:, self.s["com_vel"]:self.s["com_vel"] + 3]
        hands_rel = (self.hands[idx] - self.fruit[idx, None]).reshape(n, 6)
        priv = np.concatenate([qpos[:, 2:3], tu[:, 2:3], com_v, hands_rel, self.dr_params[idx]], 1)
        cobs = np.concatenate([clean, task, priv], 1)
        return obs.astype(np.float32), cobs.astype(np.float32), dict(q=q, qd=qd, qvel=qvel, qpos=qpos)

    def observe_all(self):
        ids = np.arange(self.N)
        obs, cobs, _ = self._observe(self._forward_sensors(ids), ids, fresh=True)
        return obs, cobs

    # ------------------------------------------------------------------ step
    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -5.0, 5.0)
        ctrl = np.clip(B.DEFAULT_POSE + REACH_ACTION_SCALE * action, self.q_lo, self.q_hi)
        ns = B.N_SUBSTEPS
        seq = np.repeat(ctrl[:, None], ns, axis=1)
        if cfg.randomize:
            k = np.arange(ns)[None, :] < self.delay[:, None]
            seq[k] = np.repeat(self.ctrl_prev[:, None], ns, axis=1)[k]
        S, SD = self.roller.rollout(self.models, self.datas, self.state, seq)
        self.state = S[:, -1].copy()
        sd = SD[:, -1].copy()
        self.ctrl_prev = ctrl
        prev = self.last_action
        self.last_action = action.copy()
        self.steps += 1
        ids = np.arange(self.N)
        self.hands = self._hands_from_sensors(sd)
        dist = self._dist(ids)
        w = cfg.rew
        terms = {"progress": w.progress * np.clip(self.prev_dist - dist, -0.2, 0.2)}
        got = dist < cfg.reach_radius
        terms["fruit"] = w.fruit * got
        if got.any():
            gi = np.nonzero(got)[0]
            self.eaten[gi] += 1
            self._spawn_fruit(gi, self.state[gi, 1:3])
            dist[gi] = self._dist(gi)
        self.prev_dist = dist
        obs, cobs, c = self._observe(sd, ids)
        tau = sd[:, self.s_tau:self.s_tau + B.N_ALL]
        qd, q = c["qd"], c["q"]
        terms["action_rate"] = w.action_rate * ((action - prev) ** 2).sum(1)
        terms["torque"] = w.torque * (tau ** 2).sum(1)
        terms["power"] = w.power * np.abs(tau * qd).sum(1)
        terms["vel_limit"] = w.vel_limit * np.maximum(0.0, np.abs(qd) - 0.9 * B.VEL_LIMIT).sum(1)
        terms["pos_limit"] = w.pos_limit * (np.maximum(0.0, q - self.q_soft_hi)
                                            + np.maximum(0.0, self.q_soft_lo - q)).sum(1)
        terms["spin"] = w.spin * (c["qvel"][:, 3:6] ** 2).sum(1)
        rew = sum(terms.values())
        timeout = self.steps >= self.max_steps
        bad = ~np.isfinite(self.state).all(1) | (np.abs(self.state).max(1) > 1e3)
        done = timeout | bad
        rew = np.where(bad, 0.0, rew)
        info = dict(terms=terms, timeout=timeout & ~bad, fallen=bad, got=got,
                    pelvis_z=c["qpos"][:, 2].copy())
        if done.any():
            dids = np.nonzero(done)[0]
            self.ep_eaten.extend(self.eaten[dids].tolist())
            self.reset(dids)
            o2, c2, _ = self._observe(self._forward_sensors(dids), dids, fresh=True)
            obs[dids], cobs[dids] = o2, c2
        return obs, cobs, rew.astype(np.float32), done, info

    def _hands_from_sensors(self, sd):
        return np.stack([sd[:, self.s_hand[0]:self.s_hand[0] + 3],
                         sd[:, self.s_hand[1]:self.s_hand[1] + 3]], 1)

    @property
    def s_hand(self):
        if not hasattr(self, "_s_hand"):
            m = self.base
            self._s_hand = [int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)])
                            for n in ("hand_l_pos", "hand_r_pos")]
        return self._s_hand

    @property
    def obs_dim(self):
        return self.cfg.history * self.PROPRIO_DIM + self.TASK_DIM

    @property
    def cobs_dim(self):
        return self.PROPRIO_DIM + self.TASK_DIM + self.PRIV_DIM

    @property
    def act_dim(self):
        return B.N_ALL
