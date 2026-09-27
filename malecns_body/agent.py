"""Closed loop: world senses -> maleCNS spiking brain -> descending neurons -> body.

Every 20 ms control tick:

1. the world is sensed (target fruit, looming logs, taste of sugar)
2. those senses become Poisson spike rates on identified sensory neurons
3. the 164k-neuron spiking model runs for 20 ms
4. descending-neuron firing is decoded into (speed, turn, jump, feed)
5. the "nerve cord" arbitrates motor programs: a Giant Fiber burst launches a
   jump, MN9 activity with food in reach starts feeding, otherwise walk/run
6. the learned locomotion controller (deploy.py) turns the command into
   joint targets for the 23 servos of the humanoid body

Hunger is the one internal state the wiring diagram cannot supply (it is
neuromodulatory). It is represented as tonic excitation of the forward-
walking command neurons (P9 / oDN1 / BDN2) and as the gain of the pursuit
channel, and it drops when the body eats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import body as B
from .brain import Brain, LIFParams
from .circuits import MotorDecoder, DecoderConfig, build_interface
from .deploy import LocomotionController, MujocoBody
from .world import World


@dataclass
class AgentConfig:
    pursuit_hz: float = 90.0          # LC10 drive for a salient target at full hunger
    loom_hz: float = 160.0            # LPLC2/LC4 drive at full looming
    loom_ref: float = 0.6             # rad/s of angular expansion for full drive
    loom_min: float = 0.12            # rad/s below which looming is not signalled
    sugar_hz: float = 100.0
    hunger_hz: float = 55.0           # tonic drive of forward-walking neurons at hunger 1
    hunger_rate: float = 0.004        # hunger increase per second
    meal: float = 0.45                # hunger removed per fruit
    feed_duration: float = 2.2        # s
    jump_cooldown: float = 1.4        # s
    vx_slew: float = 2.5              # m/s^2
    yaw_slew: float = 4.0             # rad/s^2


class FlyBrainedHumanoid:
    def __init__(self, conn, policy_path: str, world: World, cfg: AgentConfig | None = None,
                 lif: LIFParams | None = None, decoder: DecoderConfig | None = None, seed: int = 0):
        self.cfg = cfg or AgentConfig()
        self.world = world
        self.conn = conn
        self.iface = build_interface(conn)
        self.brain = Brain(conn, lif, seed=seed)
        self.decoder = MotorDecoder(self.iface, decoder)
        self.body = MujocoBody(world.model, world.data)
        self.body.reset()
        self.ctl = LocomotionController(policy_path)
        self.ctl.set_joint_limits(self.body.lo, self.body.hi)
        self.fwd_idx = np.concatenate(self.iface.readout["FORWARD"])
        self.hunger = world.sc.hunger0
        self.vx = 0.0
        self.yaw = 0.0
        self.feed_t = -1.0
        self.feed_fruit = None
        self.jump_cool = 0.0
        self.t = 0.0
        self.behaviour = "stand"
        self.inputs = {}
        self.last_counts = np.zeros(conn.n, np.int64)
        self.events = []
        self.fallen = False

    # ------------------------------------------------------------------
    def _drive_senses(self, s):
        c, I, br = self.cfg, self.iface, self.brain
        br.clear_inputs()
        pl = pr = 0.0
        if s["target"] is not None and self.feed_t < 0:
            _, az, alpha, _ = s["target"]
            sal = np.clip(alpha / np.deg2rad(6.0), 0.25, 1.0)
            gain = c.pursuit_hz * (0.3 + 0.7 * self.hunger) * sal
            wl = 1.0 / (1.0 + np.exp(-az / 0.35))   # left eye sees the left hemifield
            pl, pr = gain * wl, gain * (1 - wl)
            br.set_rates(I.pursuit_l, pl)
            br.set_rates(I.pursuit_r, pr)
        f = lambda x: c.loom_hz * np.clip((x - c.loom_min) / (c.loom_ref - c.loom_min), 0.0, 1.0)
        ll, lr = f(s["loom_l"]), f(s["loom_r"])
        if ll > 0:
            br.set_rates(I.loom_l, ll)
        if lr > 0:
            br.set_rates(I.loom_r, lr)
        sugar = c.sugar_hz if (s["taste"] or self.feed_t >= 0) else 0.0
        if sugar > 0:
            br.set_rates(I.sugar, sugar)
        hz = c.hunger_hz * self.hunger
        if hz > 0:
            br.set_rates(self.fwd_idx, hz)
        self.inputs = dict(pursuit_l=pl, pursuit_r=pr, loom_l=ll, loom_r=lr, sugar=sugar, hunger_drive=hz)

    def step(self):
        c, dt, w = self.cfg, B.CONTROL_DT, self.world
        s = w.sense()
        self._drive_senses(s)
        counts = self.brain.run(dt * 1000.0)
        self.last_counts = counts
        cmd = self.decoder.update(counts, dt * 1000.0)
        self.cmd = cmd

        # ---- nerve-cord arbitration of motor programs
        jump = False
        self.jump_cool -= dt
        vx_t, yaw_t, feed_level = cmd["vx"], cmd["yaw"], 0.0
        if self.feed_t >= 0:
            self.feed_t += dt
            vx_t, yaw_t = 0.0, 0.0
            feed_level = float(np.clip(self.feed_t / 0.6, 0, 1) * np.clip((c.feed_duration - self.feed_t) / 0.5, 0, 1))
            if self.feed_t > c.feed_duration - 0.5 and not w.fruits[self.feed_fruit].eaten:
                fr = w.fruits[self.feed_fruit]
                fr.eaten, fr.held = True, False
                w.data.mocap_pos[w.fruit_mocap[self.feed_fruit]] = [0, 0, -5]
                self.hunger = max(0.05, self.hunger - c.meal)
                self.events.append((self.t, f"ate fruit {self.feed_fruit} (hunger -> {self.hunger:.2f})"))
            if self.feed_t >= c.feed_duration:
                self.feed_t, self.feed_fruit = -1.0, None
            self.behaviour = "eat"
        elif cmd["jump"] and self.jump_cool <= 0 and not self.ctl.jumping:
            jump = True
            self.jump_cool = c.jump_cooldown
            self.events.append((self.t, f"Giant Fiber burst {self.decoder.rates['GF'].max():.0f} Hz -> JUMP"))
        elif cmd["feed"] and s["reach_fruit"] is not None:
            self.feed_t, self.feed_fruit = 0.0, s["reach_fruit"]
            w.fruits[self.feed_fruit].held = True
            self.events.append((self.t, f"MN9 {self.decoder.rates['FEED'].max():.0f} Hz with fruit in reach -> EAT"))
        if self.feed_t < 0:
            if self.ctl.jumping or jump:
                self.behaviour = "jump"
            elif abs(self.vx) < 0.15 and abs(self.yaw) < 0.15:
                self.behaviour = "stand"
            elif self.vx > 2.0:
                self.behaviour = "run"
            elif self.vx < -0.1:
                self.behaviour = "back up"
            else:
                self.behaviour = "walk"

        self.vx += np.clip(vx_t - self.vx, -c.vx_slew * dt, c.vx_slew * dt)
        self.yaw += np.clip(yaw_t - self.yaw, -c.yaw_slew * dt, c.yaw_slew * dt)

        quat, gyro = self.body.read_imu()
        q, dq = self.body.read_joints()
        target = self.ctl.step(quat, gyro, q, dq, self.vx, self.yaw, jump=jump, feed=feed_level)
        self.body.command_joints(target)
        self.body.advance()
        w.update(dt)
        self.hunger = min(1.0, self.hunger + c.hunger_rate * dt)
        self.t += dt
        if w.data.qpos[2] < 0.5 and not self.fallen:
            self.fallen = True
            self.events.append((self.t, "fell"))
        return cmd
