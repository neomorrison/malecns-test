"""Closed loop: world senses -> decision maker -> nerve cord -> learned body.

Every 20 ms control tick:

1. the world is sensed (target fruit, looming logs, taste of sugar)
2. a controller turns senses + hunger into a command (speed, turn, jump, feed):
   the maleCNS spiking brain, a trained RL model, or hand-coded rules
   (see controllers.py)
3. the "nerve cord" arbitrates motor programs: a jump command launches the
   jump manoeuvre, a feed command with food in reach starts the feeding
   program, otherwise walk / run / turn
4. the learned locomotion controller (deploy.py) turns the command into joint
   targets for the 23 servos of the humanoid body

Hunger rises slowly and drops when the body eats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import body as B
from .controllers import Controller, FlyBrainController
from .deploy import LocomotionController, MujocoBody
from .world import World


@dataclass
class AgentConfig:
    hunger_rate: float = 0.004        # hunger increase per second
    meal: float = 0.45                # hunger removed per fruit
    feed_duration: float = 2.2        # s
    jump_cooldown: float = 1.4        # s
    vx_slew: float = 2.5              # m/s^2
    yaw_slew: float = 4.0             # rad/s^2
    run_yaw_scale: float = 0.5        # sharp turns are not possible at a sprint


class Agent:
    def __init__(self, world: World, policy_path: str, controller: Controller, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.world = world
        self.controller = controller
        self.body = MujocoBody(world.model, world.data)
        self.body.reset()
        self.ctl = LocomotionController(policy_path)
        self.ctl.set_joint_limits(self.body.lo, self.body.hi)
        self.hunger = world.sc.hunger0
        self.vx = self.yaw = 0.0
        self.feed_t = -1.0
        self.feed_fruit = None
        self.jump_cool = 0.0
        self.t = 0.0
        self.behaviour = "stand"
        self.events = []
        self.fallen = False
        self.cmd = dict(vx=0.0, yaw=0.0, jump=False, feed=False)
        self.stats = dict(eaten=[], jumps=0, falls=0, distance=0.0)

    # convenience for the fly-brain video panel
    @property
    def conn(self):
        return self.controller.conn

    def step(self):
        c, dt, w = self.cfg, B.CONTROL_DT, self.world
        s = w.sense()
        cmd = self.controller.decide(s, self.hunger, self.feed_t >= 0)
        self.cmd = cmd

        jump = False
        self.jump_cool -= dt
        vx_t, yaw_t, feed_level = cmd["vx"], cmd["yaw"], 0.0
        if self.feed_t >= 0:
            self.feed_t += dt
            vx_t, yaw_t = 0.0, 0.0
            feed_level = float(np.clip(self.feed_t / 0.6, 0, 1) * np.clip((c.feed_duration - self.feed_t) / 0.5, 0, 1))
            fr = w.fruits[self.feed_fruit]
            if self.feed_t > c.feed_duration - 0.5 and not fr.eaten:
                fr.eaten, fr.held = True, False
                w.data.mocap_pos[w.fruit_mocap[self.feed_fruit]] = [0, 0, -5]
                self.hunger = max(0.05, self.hunger - c.meal)
                self.stats["eaten"].append(round(self.t, 2))
                self.events.append((self.t, f"ate fruit {self.feed_fruit} (hunger -> {self.hunger:.2f})"))
            if self.feed_t >= c.feed_duration:
                self.feed_t, self.feed_fruit = -1.0, None
            self.behaviour = "eat"
        elif cmd["jump"] and self.jump_cool <= 0 and not self.ctl.jumping and not self.fallen:
            jump = True
            self.jump_cool = c.jump_cooldown
            self.stats["jumps"] += 1
            self.events.append((self.t, "JUMP"))
        elif cmd["feed"] and s["reach_fruit"] is not None and not self.fallen:
            self.feed_t, self.feed_fruit = 0.0, s["reach_fruit"]
            w.fruits[self.feed_fruit].held = True
            self.events.append((self.t, "EAT"))
        if self.feed_t < 0:
            if self.fallen:
                self.behaviour = "fallen"
            elif self.ctl.jumping or jump:
                self.behaviour = "jump"
            elif abs(self.vx) < 0.15 and abs(self.yaw) < 0.15:
                self.behaviour = "stand"
            elif self.vx > 2.0:
                self.behaviour = "run"
            elif self.vx < -0.1:
                self.behaviour = "back up"
            else:
                self.behaviour = "walk"

        if vx_t > 2.0:
            yaw_t *= c.run_yaw_scale
        self.vx += np.clip(vx_t - self.vx, -c.vx_slew * dt, c.vx_slew * dt)
        self.yaw += np.clip(yaw_t - self.yaw, -c.yaw_slew * dt, c.yaw_slew * dt)

        p0 = w.data.qpos[:2].copy()
        quat, gyro = self.body.read_imu()
        q, dq = self.body.read_joints()
        target = self.ctl.step(quat, gyro, q, dq, self.vx, self.yaw, jump=jump, feed=feed_level)
        self.body.command_joints(target)
        self.body.advance(callback=w.check_log_contacts)
        w.update(B.CONTROL_DT)
        self.stats["distance"] += float(np.linalg.norm(w.data.qpos[:2] - p0))
        self.hunger = min(1.0, self.hunger + c.hunger_rate * dt)
        self.t += dt
        if w.data.qpos[2] < 0.5 and not self.fallen:
            self.fallen = True
            self.stats["falls"] += 1
            self.events.append((self.t, "fell"))
        return cmd


class FlyBrainedHumanoid(Agent):
    """The humanoid driven by the maleCNS spiking brain."""

    def __init__(self, conn, policy_path: str, world: World, cfg: AgentConfig | None = None,
                 decoder=None, seed: int = 0, **kw):
        super().__init__(world, policy_path, FlyBrainController(conn, decoder=decoder, seed=seed, **kw), cfg)

    @property
    def iface(self):
        return self.controller.iface

    @property
    def decoder(self):
        return self.controller.decoder

    @property
    def inputs(self):
        return self.controller.inputs

    @property
    def last_counts(self):
        return self.controller.last_counts
