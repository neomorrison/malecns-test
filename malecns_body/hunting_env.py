"""Fast abstract hunting world, used to train the "regular model" decision maker.

The learned controller (controllers.LearnedController) must see the same senses
and emit the same commands as the fly brain. Training it on the full humanoid
would be slow, so it learns here, where the body is abstracted to what the
decision layer can observe of it:

* speed and turn rate follow the command with the same slew limits the agent
  applies, plus the lag of the locomotion controller
* a jump lasts 0.9 s and the body is airborne from 0.27 s to 0.56 s; a log that
  reaches the body while it is on the ground trips it (3 s lost)
* eating takes 2.2 s standing still and removes 0.45 hunger

Senses are computed exactly like World.sense (target azimuth / angular size,
looming rate per eye, taste within reach). Vectorised over many worlds.
"""
from __future__ import annotations

import numpy as np

from . import body as B
from .controllers import N_FEATURES, V_MAX, YAW_MAX
from .world import FRUIT_R, LOG_HALF_LEN, LOG_R

DT = B.CONTROL_DT * 2          # 25 Hz decisions in the abstract world


class HuntingEnv:
    def __init__(self, n: int, seed: int = 0, n_fruit: int = 5, n_prey: int = 2, episode_s: float = 60.0):
        self.N, self.rng = n, np.random.default_rng(seed)
        self.nf, self.np_ = n_fruit, n_prey
        self.max_steps = int(episode_s / DT)
        self.episode_s = episode_s
        self.obs_dim = N_FEATURES
        self.cobs_dim = N_FEATURES + 6
        self.act_dim = 4
        N = n
        self.pos = np.zeros((N, 2))
        self.yawang = np.zeros(N)
        self.v = np.zeros(N)
        self.w = np.zeros(N)
        self.cmd = np.zeros((N, 2))
        self.hunger = np.zeros(N)
        self.fruit = np.zeros((N, n_fruit, 2))
        self.fvel = np.zeros((N, n_fruit, 2))
        self.eaten = np.zeros((N, n_fruit), bool)
        self.flee = np.zeros((N, n_fruit))
        self.flee_r = np.zeros((N, n_fruit))
        self.stamina = np.zeros((N, n_fruit))
        self.log_pos = np.zeros((N, 2))
        self.log_vel = np.zeros((N, 2))
        self.log_active = np.zeros(N, bool)
        self.log_hit = np.zeros(N, bool)
        self.next_log = np.zeros(N)
        self.jump_t = np.full(N, -1.0)
        self.jump_cool = np.zeros(N)
        self.feed_t = np.full(N, -1.0)
        self.feed_i = np.zeros(N, int)
        self.trip_t = np.zeros(N)
        self.t = np.zeros(N)
        self.steps = np.zeros(N, int)
        self.stats = []
        self.reset(np.arange(N))

    def reset(self, idx):
        r, n = self.rng, len(idx)
        self.pos[idx] = 0
        self.yawang[idx] = 0
        self.v[idx] = self.w[idx] = 0
        self.cmd[idx] = 0
        self.hunger[idx] = 0.95
        d = r.uniform(6, 20, (n, self.nf))
        a = r.uniform(-1.8, 1.8, (n, self.nf))
        self.fruit[idx] = np.stack([d * np.cos(a), d * np.sin(a)], -1)
        self.fvel[idx] = 0
        self.eaten[idx] = False
        fl = np.zeros((n, self.nf))
        fl[:, :self.np_] = r.uniform(1.8, 2.4, (n, self.np_))
        self.flee[idx] = fl
        self.flee_r[idx] = r.uniform(5, 7, (n, self.nf))
        self.stamina[idx] = r.uniform(4, 7, (n, self.nf))
        self.log_active[idx] = False
        self.log_hit[idx] = False
        self.next_log[idx] = r.uniform(8, 20, n)
        self.jump_t[idx] = -1
        self.jump_cool[idx] = 0
        self.feed_t[idx] = -1
        self.trip_t[idx] = 0
        self.t[idx] = 0
        self.steps[idx] = 0

    # ------------------------------------------------------------------ senses
    def _senses(self):
        N = self.N
        eye = self.pos
        rel = self.fruit - eye[:, None]
        dist = np.linalg.norm(rel, axis=2) + 1e-6
        az = (np.arctan2(rel[..., 1], rel[..., 0]) - self.yawang[:, None] + np.pi) % (2 * np.pi) - np.pi
        alpha = 2 * np.arctan(FRUIT_R / np.sqrt(dist ** 2 + 1.5 ** 2))
        vis = (~self.eaten) & (np.abs(az) < np.deg2rad(160)) & (dist < 30)
        score = np.where(vis, alpha, -1)
        best = score.argmax(1)
        ar = np.arange(N)
        has = score[ar, best] > 0
        taste = ((dist < 0.8) & ~self.eaten).any(1)
        reach = np.where(((dist < 0.8) & ~self.eaten).any(1), ((dist < 0.8) & ~self.eaten).argmax(1), -1)
        # looming of the log (object's own approach)
        lr = self.log_pos - eye
        ld = np.linalg.norm(lr, axis=1) + 1e-6
        v_app = -(self.log_vel * lr).sum(1) / ld
        size = LOG_R + 0.3 * LOG_HALF_LEN
        ad = np.where(self.log_active & (v_app > 0), 2 * size * v_app / (ld ** 2 + size ** 2), 0.0)
        laz = (np.arctan2(lr[:, 1], lr[:, 0]) - self.yawang + np.pi) % (2 * np.pi) - np.pi
        side_l = 1 / (1 + np.exp(-laz / 0.2))
        return dict(has=has, az=az[ar, best], alpha=alpha[ar, best], dist=dist[ar, best],
                    loom_l=ad * side_l, loom_r=ad * (1 - side_l), taste=taste, reach=reach, log_dist=ld)

    def _features(self, s):
        vis = np.where(s["has"][:, None],
                       np.stack([np.ones(self.N), np.sin(s["az"]), np.cos(s["az"]), np.minimum(s["alpha"] / 0.1, 3)], 1),
                       np.array([0.0, 0.0, 1.0, 0.0]))
        return np.concatenate([vis, np.minimum(s["loom_l"], 3)[:, None], np.minimum(s["loom_r"], 3)[:, None],
                               s["taste"][:, None].astype(float), self.hunger[:, None],
                               self.cmd / np.array([V_MAX, YAW_MAX])], 1).astype(np.float32)

    def observe_all(self):
        s = self._senses()
        o = self._features(s)
        return o, self._critic(o, s)

    def _critic(self, o, s):
        return np.concatenate([o, np.stack([s["dist"] / 20, np.minimum(s["log_dist"], 20) / 20,
                                            self.v / V_MAX, self.w, (self.feed_t >= 0) * 1.0,
                                            (self.jump_t >= 0) * 1.0], 1)], 1).astype(np.float32)

    # -------------------------------------------------------------------- step
    def step(self, a):
        N, r = self.N, self.rng
        a = np.asarray(a, dtype=np.float64)
        vx_c = V_MAX / (1 + np.exp(-a[:, 0]))
        yaw_c = YAW_MAX * np.tanh(a[:, 1])
        jump_c, feed_c = a[:, 2] > 0, a[:, 3] > 0
        s0 = self._senses()
        rew = np.zeros(N)
        # motor programs (same arbitration as agent.Agent)
        feeding = self.feed_t >= 0
        tripped = self.trip_t > 0
        start_jump = jump_c & (self.jump_cool <= 0) & (self.jump_t < 0) & ~feeding & ~tripped
        self.jump_t[start_jump] = 0.0
        self.jump_cool[start_jump] = 1.4
        start_feed = feed_c & (s0["reach"] >= 0) & ~feeding & ~start_jump & ~tripped
        self.feed_t[start_feed] = 0.0
        self.feed_i[start_feed] = s0["reach"][start_feed]
        rew -= 0.05 * start_jump            # jumping costs energy
        vx_t = np.where((self.feed_t >= 0) | tripped, 0.0, vx_c)
        yaw_t = np.where((self.feed_t >= 0) | tripped, 0.0, np.where(vx_t > 2.0, 0.5 * yaw_c, yaw_c))
        self.cmd = np.stack([vx_c, yaw_c], 1)
        # body: the agent's slew-limited command
        self.v += np.clip(vx_t - self.v, -2.5 * DT, 2.5 * DT)
        self.w += np.clip(yaw_t - self.w, -4.0 * DT, 4.0 * DT)
        self.yawang += self.w * DT
        self.pos += self.v[:, None] * DT * np.stack([np.cos(self.yawang), np.sin(self.yawang)], 1)
        # jumps / feeding / trips advance
        self.jump_cool -= DT
        j = self.jump_t >= 0
        self.jump_t[j] += DT
        self.jump_t[self.jump_t > 0.9] = -1.0
        self.trip_t = np.maximum(0, self.trip_t - DT)
        f = self.feed_t >= 0
        self.feed_t[f] += DT
        done_feed = f & (self.feed_t >= 1.7) & ~self.eaten[np.arange(N), self.feed_i]
        if done_feed.any():
            ids = np.nonzero(done_feed)[0]
            self.eaten[ids, self.feed_i[ids]] = True
            rew[ids] += 10.0 * self.hunger[ids]
            self.hunger[ids] = np.maximum(0.05, self.hunger[ids] - 0.45)
        self.feed_t[self.feed_t >= 2.2] = -1.0
        # prey behaviour (same as World.update)
        rel = self.fruit - self.pos[:, None]
        dist = np.linalg.norm(rel, axis=2)
        fleeing = (self.flee > 0) & (dist < self.flee_r) & (self.stamina > 0) & ~self.eaten
        away = np.arctan2(rel[..., 1], rel[..., 0]) + 0.5 * np.sin(1.3 * self.t)[:, None] + r.normal(0, 0.3, dist.shape)
        sp = self.flee * np.clip(self.stamina / 2.0, 0.25, 1.0)
        self.fvel = np.where(fleeing[..., None], sp[..., None] * np.stack([np.cos(away), np.sin(away)], -1), self.fvel * 0.9)
        self.stamina -= fleeing * DT
        self.fruit += self.fvel * DT
        # logs: spawn ahead and roll at the body
        spawn = (~self.log_active) & (self.t >= self.next_log)
        if spawn.any():
            ids = np.nonzero(spawn)[0]
            ahead = r.uniform(6.5, 8.0, len(ids))
            sp_ = r.uniform(2.5, 3.5, len(ids))
            fwd = np.stack([np.cos(self.yawang[ids]), np.sin(self.yawang[ids])], 1)
            self.log_pos[ids] = self.pos[ids] + ahead[:, None] * fwd
            self.log_vel[ids] = -sp_[:, None] * fwd
            self.log_active[ids] = True
            self.log_hit[ids] = False
        self.log_pos += self.log_vel * DT * self.log_active[:, None]
        lrel = self.pos - self.log_pos
        along = (lrel * self.log_vel).sum(1) / (np.linalg.norm(self.log_vel, axis=1) + 1e-6)
        perp = np.abs(lrel[:, 0] * self.log_vel[:, 1] - lrel[:, 1] * self.log_vel[:, 0]) / (np.linalg.norm(self.log_vel, axis=1) + 1e-6)
        at_body = self.log_active & (np.abs(along) < LOG_R + 0.15) & (perp < LOG_HALF_LEN + 0.2)
        airborne = (self.jump_t > 0.27) & (self.jump_t < 0.56)
        hit = at_body & ~airborne & ~self.log_hit
        if hit.any():
            self.log_hit[hit] = True
            self.trip_t[hit] = 3.0
            self.jump_t[hit] = -1.0
            self.feed_t[hit] = -1.0
            rew[hit] -= 3.0
        passed = self.log_active & (along < -1.0)
        cleared = passed & ~self.log_hit
        rew[cleared] += 1.0
        self.log_active[passed] = False
        self.next_log[passed] = self.t[passed] + r.uniform(6, 14, passed.sum())
        # hunger
        self.hunger = np.minimum(1.0, self.hunger + 0.004 * DT)
        rew -= 0.02 * self.hunger * DT / 0.04
        self.t += DT
        self.steps += 1
        timeout = self.steps >= self.max_steps
        done = timeout.copy()
        s = self._senses()
        o = self._features(s)
        c = self._critic(o, s)
        info = dict(terms=dict(eat=rew * 0), timeout=timeout, fallen=np.zeros(N, bool))
        if done.any():
            ids = np.nonzero(done)[0]
            for i in ids:
                self.stats.append(int(self.eaten[i].sum()))
            self.reset(ids)
            s2 = self._senses()
            o2 = self._features(s2)
            o[ids], c[ids] = o2[ids], self._critic(o2, s2)[ids]
        return o, c, rew.astype(np.float32), done, info
