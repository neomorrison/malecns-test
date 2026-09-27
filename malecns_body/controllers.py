"""Three ways to decide what the body does, behind one interface.

Every controller sees the same senses (from ``World.sense``) plus the body's
hunger, and returns the same command: forward speed, turn rate, jump, feed.
The shared agent (agent.py) then runs the same motor-program arbitration and
the same learned locomotion controller, so any difference in behaviour comes
from the decision maker alone.

* ``FlyBrainController``  - the maleCNS spiking brain (141,615 neurons)
* ``LearnedController``   - a small neural network trained with RL to hunt
* ``HandCodedController`` - classic robotics rules
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from . import body as B
from .brain import Brain, LIFParams
from .circuits import DecoderConfig, MotorDecoder, build_interface

V_MAX = 3.5
YAW_MAX = 1.0


class Controller:
    name = "controller"

    def decide(self, s: dict, hunger: float, feeding: bool) -> dict:
        raise NotImplementedError

    def reset(self):
        pass


# --------------------------------------------------------------------------
# the fly brain
# --------------------------------------------------------------------------

@dataclass
class FlyBrainConfig:
    pursuit_hz: float = 90.0          # LC10 drive for a salient target at full hunger
    loom_hz: float = 160.0            # LPLC2/LC4 drive at full looming
    loom_ref: float = 0.6             # rad/s of angular expansion for full drive
    loom_min: float = 0.12            # rad/s below which looming is not signalled
    sugar_hz: float = 100.0
    hunger_hz: float = 55.0           # tonic drive of forward-walking neurons at hunger 1


class FlyBrainController(Controller):
    """Senses -> Poisson spikes on identified neurons -> spiking maleCNS -> decoded DNs."""
    name = "maleCNS fly brain"

    def __init__(self, conn, cfg: FlyBrainConfig | None = None, lif: LIFParams | None = None,
                 decoder: DecoderConfig | None = None, seed: int = 0):
        self.cfg = cfg or FlyBrainConfig()
        self.conn = conn
        self.iface = build_interface(conn)
        self.brain = Brain(conn, lif, seed=seed)
        self.decoder = MotorDecoder(self.iface, decoder)
        self.fwd_idx = np.concatenate(self.iface.readout["FORWARD"])
        self.inputs = {}
        self.last_counts = np.zeros(conn.n, np.int64)
        self.compute_s = 0.0

    def _drive(self, s, hunger, feeding):
        c, I, br = self.cfg, self.iface, self.brain
        br.clear_inputs()
        pl = pr = 0.0
        if s["target"] is not None and not feeding:
            _, az, alpha, _ = s["target"]
            sal = np.clip(alpha / np.deg2rad(6.0), 0.25, 1.0)
            gain = c.pursuit_hz * (0.3 + 0.7 * hunger) * sal
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
        sugar = c.sugar_hz if (s["taste"] or feeding) else 0.0
        if sugar > 0:
            br.set_rates(I.sugar, sugar)
        hz = c.hunger_hz * hunger
        if hz > 0:
            br.set_rates(self.fwd_idx, hz)
        self.inputs = dict(pursuit_l=pl, pursuit_r=pr, loom_l=ll, loom_r=lr, sugar=sugar, hunger_drive=hz)

    def decide(self, s, hunger, feeding):
        t0 = time.perf_counter()
        self._drive(s, hunger, feeding)
        counts = self.brain.run(B.CONTROL_DT * 1000.0)
        self.last_counts = counts
        cmd = self.decoder.update(counts, B.CONTROL_DT * 1000.0)
        self.compute_s += time.perf_counter() - t0
        return cmd


# --------------------------------------------------------------------------
# a trained neural network
# --------------------------------------------------------------------------

def hunting_features(s, hunger, prev_cmd):
    """Observation vector shared by the RL hunting env and LearnedController."""
    tgt = s["target"]
    if tgt is None:
        vis = [0.0, 0.0, 1.0, 0.0]
    else:
        _, az, alpha, _ = tgt
        vis = [1.0, np.sin(az), np.cos(az), min(alpha / 0.1, 3.0)]
    return np.array(vis + [min(s["loom_l"], 3.0), min(s["loom_r"], 3.0), float(s["taste"]),
                           hunger, prev_cmd[0] / V_MAX, prev_cmd[1] / YAW_MAX], np.float32)


N_FEATURES = 10


def hunting_action_to_command(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(vx=float(V_MAX / (1.0 + np.exp(-a[0]))), yaw=float(YAW_MAX * np.tanh(a[1])),
                jump=bool(a[2] > 0.0), feed=bool(a[3] > 0.0))


class LearnedController(Controller):
    """MLP trained with PPO in the abstract hunting env (scripts/train_hunter.py)."""
    name = "trained RL model"

    def __init__(self, path):
        z = np.load(path)
        self.nl = int(z["n_layers"])
        self.W = [z[f"W{i}"].astype(np.float64) for i in range(self.nl)]
        self.b = [z[f"b{i}"].astype(np.float64) for i in range(self.nl)]
        self.mean, self.std, self.clip = z["obs_mean"], z["obs_std"], float(z["obs_clip"])
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        self.n_params = sum(w.size for w in self.W) + sum(b.size for b in self.b)
        self.prev = np.zeros(2)
        self.compute_s = 0.0

    def reset(self):
        self.prev = np.zeros(2)

    def decide(self, s, hunger, feeding):
        t0 = time.perf_counter()
        x = hunting_features(s, hunger, self.prev)
        x = np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        for i in range(self.nl - 1):
            x = x @ self.W[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        a = x @ self.W[-1] + self.b[-1]
        cmd = hunting_action_to_command(a)
        self.prev = np.array([cmd["vx"], cmd["yaw"]])
        self.compute_s += time.perf_counter() - t0
        return cmd


# --------------------------------------------------------------------------
# hand-written rules
# --------------------------------------------------------------------------

class HandCodedController(Controller):
    """Proportional pursuit, jump on fast looming, eat when food is in reach."""
    name = "hand-coded rules"

    def __init__(self, k_turn=2.0, loom_jump=0.35):
        self.k_turn, self.loom_jump = k_turn, loom_jump
        self.compute_s = 0.0

    def decide(self, s, hunger, feeding):
        t0 = time.perf_counter()
        vx = yaw = 0.0
        if s["target"] is not None:
            _, az, _, dist = s["target"]
            yaw = float(np.clip(self.k_turn * az, -YAW_MAX, YAW_MAX))
            vx = V_MAX * np.clip(hunger, 0, 1) * np.clip(np.cos(az), 0.2, 1.0) * np.clip(dist / 2.0, 0.3, 1.0)
        cmd = dict(vx=float(vx), yaw=yaw, jump=bool(max(s["loom_l"], s["loom_r"]) > self.loom_jump),
                   feed=bool(s["taste"]))
        self.compute_s += time.perf_counter() - t0
        return cmd
