"""Sensory inputs and motor readouts: where the body meets the connectome.

Inputs are identified sensory / visual projection neurons of maleCNS;
readouts are identified descending neurons (the ~1,300 cells carrying
commands from the brain to the ventral nerve cord) plus the proboscis motor
neuron MN9. Each choice cites how the cell type is known to act in the fly
and was checked with ``scripts/probe_brain.py`` (see results/brain_probe.md).

The spiking model decides which readouts fire. ``MotorDecoder`` only
translates descending-neuron firing into the body's command interface
(speed, turn, jump, feed) - the job the fly's nerve cord does for its legs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

# --------------------------------------------------------------------------
# sensory inputs
# --------------------------------------------------------------------------
# Sugar gustatory receptor neurons of the labellum. Identified from the
# wiring: LB3a-d are the taste neurons that synapse directly onto the sugar
# second-order neurons Bract and Usnea (Shiu et al. 2022, eLife 11:e79887),
# and in the spiking model they drive the feeding motor neuron MN9.
SUGAR_GRN_TYPES = ["LB3a", "LB3b", "LB3c", "LB3d"]

# Visual object-tracking neurons. LC10 cells project to the anterior optic
# tubercle and are how a male tracks and chases a target during courtship
# (Ribeiro et al. 2018; Hindmarsh Sten et al. 2021). In the model, LC10a/c-1/d/e
# on one side excite the steering neurons of the *same* side (turn towards
# the target), and LC10e/c-1 also excite forward-walking neurons.
PURSUIT_TYPES = ["LC10a", "LC10c-1", "LC10d", "LC10e"]

# Looming detectors: LPLC2 and LC4, the visual inputs of the Giant Fiber
# escape neuron (von Reyn et al. 2017; Ache et al. 2019).
LOOM_TYPES = ["LPLC2", "LC4"]

# Food-odour olfactory receptor neurons (DM1/Or42b, DM2/Or22a, DM4/Or59b,
# VA2/Or92a; Semmelhack & Wang 2009). Probed, but NOT used by the agent: in
# this point-neuron model odour drives a self-sustaining antennal-lobe /
# mushroom-body state and no reliable steering (see results/brain_probe.md).
FOOD_ORN_TYPES = ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VA2"]

# --------------------------------------------------------------------------
# motor readouts
# --------------------------------------------------------------------------
READOUTS = {
    # escape take-off / jump: the Giant Fiber (von Reyn et al. 2014)
    "GF": ["DNp01"],
    # steering: DNa02, DNa01 and DNg13 activity predicts ipsilateral turning
    # (Rayshubskiy et al. 2020; Yang et al. 2023)
    "STEER": ["DNa02", "DNa01", "DNg13"],
    # forward walking: P9 = DNp09 (Bidaye et al. 2020), oDN1 = DNg97 and
    # BDN2 = DNg100 (Sapkal et al. 2024)
    "FORWARD": ["DNp09", "DNg97", "DNg100"],
    # backward walking: moonwalker descending neuron (Bidaye et al. 2014)
    "BACKWARD": ["MDN"],
    # proboscis extension / feeding motor neuron (Shiu et al. 2022/2024)
    "FEED": ["MN9"],
}


@dataclass
class Interface:
    """Neuron indices for every input channel and readout group."""
    sugar: np.ndarray
    pursuit_l: np.ndarray
    pursuit_r: np.ndarray
    loom_l: np.ndarray
    loom_r: np.ndarray
    odor_l: np.ndarray
    odor_r: np.ndarray
    readout: dict = field(default_factory=dict)   # name -> (idx_L, idx_R)

    def summary(self):
        lines = [f"sugar GRNs        {len(self.sugar):4d}        ({', '.join(SUGAR_GRN_TYPES)})",
                 f"pursuit VPNs L/R  {len(self.pursuit_l):4d} / {len(self.pursuit_r):<4d} ({', '.join(PURSUIT_TYPES)})",
                 f"loom VPNs    L/R  {len(self.loom_l):4d} / {len(self.loom_r):<4d} ({', '.join(LOOM_TYPES)})",
                 f"food ORNs    L/R  {len(self.odor_l):4d} / {len(self.odor_r):<4d} ({', '.join(FOOD_ORN_TYPES)})"]
        for k, (l, r) in self.readout.items():
            lines.append(f"readout {k:9s} L/R {len(l)}/{len(r)}  ({', '.join(READOUTS[k])})")
        return "\n".join(lines)


def build_interface(conn: Connectome) -> Interface:
    ro = {k: (conn.select(t, side="L"), conn.select(t, side="R")) for k, t in READOUTS.items()}
    return Interface(
        sugar=conn.select(SUGAR_GRN_TYPES),
        pursuit_l=conn.select(PURSUIT_TYPES, side="L"),
        pursuit_r=conn.select(PURSUIT_TYPES, side="R"),
        loom_l=conn.select(LOOM_TYPES, side="L"),
        loom_r=conn.select(LOOM_TYPES, side="R"),
        odor_l=conn.select(FOOD_ORN_TYPES, side="L", side_col="rootSide"),
        odor_r=conn.select(FOOD_ORN_TYPES, side="R", side_col="rootSide"),
        readout=ro,
    )


def group_rates(iface: Interface, counts: np.ndarray, window_ms: float):
    """Mean firing rate (Hz) of each readout group, per side."""
    out = {}
    for k, (l, r) in iface.readout.items():
        out[k] = (counts[l].mean() * 1000 / window_ms if len(l) else 0.0,
                  counts[r].mean() * 1000 / window_ms if len(r) else 0.0)
    return out


@dataclass
class DecoderConfig:
    tau_ms: float = 120.0         # smoothing of readout rates
    v_max: float = 3.5            # m/s at full forward drive
    fwd_lo: float = 8.0           # Hz: forward-neuron rate at which walking starts
    fwd_hi: float = 70.0          # Hz: rate for top speed
    steer_scale: float = 60.0     # Hz of (left - right) steering-neuron rate per unit turn
    yaw_max: float = 1.0          # rad/s
    gf_jump: float = 120.0        # Hz: Giant Fiber rate that triggers a jump
    feed_on: float = 60.0         # Hz: MN9 rate that triggers feeding
    backward_on: float = 40.0     # Hz: MDN rate that triggers backing up


class MotorDecoder:
    """Descending-neuron rates -> (vx, yaw_rate, jump, feed) body commands.

    This plays the role of the fly's ventral nerve cord: fixed, simple, and
    driven by what the named command neurons are known to do.
    """

    def __init__(self, iface: Interface, cfg: DecoderConfig | None = None):
        self.iface, self.cfg = iface, cfg or DecoderConfig()
        self.rates = {k: np.zeros(2) for k in iface.readout}

    def update(self, counts: np.ndarray, window_ms: float):
        a = 1.0 - np.exp(-window_ms / self.cfg.tau_ms)
        for k, (l, r) in self.iface.readout.items():
            now = np.array([counts[l].sum(), counts[r].sum()]) * 1000.0 / window_ms
            self.rates[k] += a * (now - self.rates[k])
        c = self.cfg
        fwd = self.rates["FORWARD"].mean() / max(1, len(self.iface.readout["FORWARD"][0]))
        speed = c.v_max * np.clip((fwd - c.fwd_lo) / (c.fwd_hi - c.fwd_lo), 0.0, 1.0)
        steer = self.rates["STEER"][0] - self.rates["STEER"][1]
        yaw = c.yaw_max * np.tanh(steer / c.steer_scale)
        if speed > 2.0:                   # sharp turns are not possible at a sprint
            yaw *= 0.5
        mdn = self.rates["BACKWARD"].max() / max(1, len(self.iface.readout["BACKWARD"][0]))
        # moonwalker neurons win only if clearly stronger than the forward-walking ones
        if mdn > max(c.backward_on, 1.5 * fwd):
            speed = -0.4
        return dict(vx=float(speed), yaw=float(yaw),
                    jump=bool(self.rates["GF"].max() > c.gf_jump),
                    feed=bool(self.rates["FEED"].max() > c.feed_on),
                    fwd_rate=float(fwd), steer=float(steer))
