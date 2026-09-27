"""The arena: fruit to eat, fruit that runs away, and logs that roll at you.

The world also computes the body's senses in the form the fly brain takes
them (see circuits.py):

* vision  - the most salient fruit in view: azimuth and angular size
            (-> LC10 pursuit neurons of the eye that sees it)
* looming - how fast an approaching log grows in each eye
            (-> LPLC2 / LC4 looming detectors)
* taste   - sugar on the hands/mouth when a fruit is within reach
            (-> labellar sugar receptor neurons)

Objects are kinematic (mocap) bodies. Fruit has no collision; logs do, so a
missed jump really trips the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import body as B

FRUIT_R = 0.11
LOG_R = 0.13
LOG_HALF_LEN = 0.7


@dataclass
class Fruit:
    pos: np.ndarray
    flee_speed: float = 0.0        # > 0: prey that runs away when approached
    flee_radius: float = 6.0
    stamina: float = 5.0           # seconds of fleeing before it tires
    eaten: bool = False
    held: bool = False
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _dir: float = 0.0


@dataclass
class Log:
    pos: np.ndarray
    vel: np.ndarray
    active: bool = False
    roll: float = 0.0


@dataclass
class Scenario:
    fruits: list
    # (time s, distance ahead m, lateral offset m, speed m/s): logs rolled at the body
    log_schedule: list = field(default_factory=list)
    hunger0: float = 0.9
    duration: float = 60.0


def default_scenario(seed=0):
    return Scenario(
        fruits=[
            Fruit(np.array([7.0, 3.5])),                                   # an easy snack, ahead-left
            Fruit(np.array([17.0, -7.0]), flee_speed=2.2, flee_radius=7.0,  # prey: runs away
                  stamina=6.0),
            Fruit(np.array([4.0, -14.0])),
        ],
        log_schedule=[(19.0, 7.0, 0.0, 3.0), (31.0, 7.5, 0.0, 3.0)],
        hunger0=0.95,
        duration=60.0,
    )


class World:
    def __init__(self, scenario: Scenario | None = None, seed: int = 0):
        self.sc = scenario or default_scenario(seed)
        self.rng = np.random.default_rng(seed)
        spec = mujoco.MjSpec.from_file(B.HUMANOID_XML)
        self.fruits = [Fruit(f.pos.copy(), f.flee_speed, f.flee_radius, f.stamina) for f in self.sc.fruits]
        for i, f in enumerate(self.fruits):
            b = spec.worldbody.add_body(name=f"fruit{i}", mocap=True, pos=[*f.pos, FRUIT_R])
            col = [0.95, 0.55, 0.05, 1] if f.flee_speed > 0 else [0.85, 0.1, 0.1, 1]
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[FRUIT_R, 0, 0], rgba=col,
                       contype=0, conaffinity=0)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012, 0.03, 0], pos=[0, 0, FRUIT_R + 0.02],
                       rgba=[0.35, 0.22, 0.1, 1], contype=0, conaffinity=0)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.05, 0.02, 0.008],
                       pos=[0.03, 0, FRUIT_R + 0.04], rgba=[0.2, 0.6, 0.15, 1], contype=0, conaffinity=0)
        self.logs = []
        for j, _ in enumerate(self.sc.log_schedule):
            b = spec.worldbody.add_body(name=f"log{j}", mocap=True, pos=[0, 0, -5])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[LOG_R, LOG_HALF_LEN, 0],
                       quat=[0.7071068, 0.7071068, 0, 0], rgba=[0.45, 0.3, 0.16, 1],
                       contype=1, conaffinity=1, friction=[0.8, 0.02, 0.01])
            self.logs.append(Log(np.array([0.0, 0.0, -5.0]), np.zeros(3)))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.fruit_mocap = [self.model.body(f"fruit{i}").mocapid[0] for i in range(len(self.fruits))]
        self.log_mocap = [self.model.body(f"log{j}").mocapid[0] for j in range(len(self.logs))]
        self.head_site = self.model.site("head").id
        self.hand_sites = [self.model.site("hand_l").id, self.model.site("hand_r").id]
        self.t = 0.0
        self._log_i = 0
        self._prev_alpha = {}
        self.events = []

    # ------------------------------------------------------------------ util
    def body_pose(self):
        d = self.data
        p = d.qpos[:3].copy()
        w, x, y, z = d.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return p, yaw

    def _eye(self):
        return self.data.site_xpos[self.head_site].copy()

    # ---------------------------------------------------------------- update
    def update(self, dt, feeding_fruit=None):
        """Advance object behaviour by dt seconds."""
        self.t += dt
        p, yaw = self.body_pose()
        for i, f in enumerate(self.fruits):
            if f.eaten:
                continue
            if f.held:
                hands = self.data.site_xpos[self.hand_sites]
                pos3 = hands.mean(0)
            else:
                if f.flee_speed > 0:
                    rel = f.pos - p[:2]
                    dist = np.linalg.norm(rel)
                    if dist < f.flee_radius and f.stamina > 0:
                        away = np.arctan2(rel[1], rel[0])
                        f._dir += self.rng.normal(0, 0.6) * dt
                        heading = away + 0.5 * np.sin(1.3 * self.t) + f._dir
                        speed = f.flee_speed * np.clip(f.stamina / 2.0, 0.25, 1.0)
                        f.vel = speed * np.array([np.cos(heading), np.sin(heading)])
                        f.stamina -= dt
                    else:
                        f.vel *= 0.9
                    f.pos = f.pos + f.vel * dt
                pos3 = np.array([*f.pos, FRUIT_R])
            self.data.mocap_pos[self.fruit_mocap[i]] = pos3
        # logs: spawn on schedule, roll towards where the body is
        while self._log_i < len(self.sc.log_schedule) and self.t >= self.sc.log_schedule[self._log_i][0]:
            _, ahead, lat, speed = self.sc.log_schedule[self._log_i]
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            left = np.array([-np.sin(yaw), np.cos(yaw)])
            start = p[:2] + ahead * fwd + lat * left
            lg = self.logs[self._log_i]
            lg.pos = np.array([*start, LOG_R])
            lg.vel = np.array([*(-speed * fwd), 0.0])
            lg.active = True
            self.events.append((self.t, f"log {self._log_i} rolls in"))
            self._log_i += 1
        for j, lg in enumerate(self.logs):
            if not lg.active:
                continue
            lg.pos = lg.pos + lg.vel * dt
            v = lg.vel[:2]
            sp = np.linalg.norm(v)
            lg.roll += sp * dt / LOG_R
            ang = np.arctan2(v[1], v[0])
            # capsule axis along the local x of the body frame after the geom quat; orient
            # the log perpendicular to its motion and spin it about its axis
            q_yaw = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
            q_roll = np.array([np.cos(lg.roll / 2), 0, np.sin(lg.roll / 2), 0])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, q_yaw, q_roll)
            self.data.mocap_pos[self.log_mocap[j]] = lg.pos
            self.data.mocap_quat[self.log_mocap[j]] = q
            if np.linalg.norm(lg.pos[:2] - p[:2]) > 25:
                lg.active = False
                self.data.mocap_pos[self.log_mocap[j]] = [0, 0, -5]

    # ----------------------------------------------------------------- senses
    def sense(self, fov=np.deg2rad(160), max_dist=30.0):
        """Visual target, looming and taste for the brain."""
        p, yaw = self.body_pose()
        eye = self._eye()
        out = dict(target=None, loom_l=0.0, loom_r=0.0, taste=False, reach_fruit=None)
        best = None
        for i, f in enumerate(self.fruits):
            if f.eaten or f.held:
                continue
            rel = np.array([*f.pos, FRUIT_R]) - eye
            d = np.linalg.norm(rel)
            az = _wrap(np.arctan2(rel[1], rel[0]) - yaw)
            if d > max_dist or abs(az) > fov:
                continue
            alpha = 2 * np.arctan(FRUIT_R / d)
            if best is None or alpha > best[2]:
                best = (i, az, alpha, d)
            if np.linalg.norm(f.pos - p[:2]) < 0.8:
                out["taste"] = True
                out["reach_fruit"] = i
        out["target"] = best
        for j, lg in enumerate(self.logs):
            if not lg.active:
                continue
            rel = lg.pos - eye
            d = np.linalg.norm(rel)
            az = _wrap(np.arctan2(rel[1], rel[0]) - yaw)
            if abs(az) > fov:
                continue
            # expansion due to the object's own approach (not the body's motion)
            v_app = -np.dot(lg.vel, rel) / max(d, 1e-6)
            if v_app <= 0:
                continue
            size = LOG_R + 0.3 * LOG_HALF_LEN      # effective radius of the elongated log
            alpha_dot = 2 * size * v_app / (d * d + size * size)
            side_l = 1.0 / (1.0 + np.exp(-az / 0.2))
            out["loom_l"] = max(out["loom_l"], alpha_dot * side_l)
            out["loom_r"] = max(out["loom_r"], alpha_dot * (1 - side_l))
        return out


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
