"""Whole-CNS spiking model of the maleCNS connectome.

Every traced neuron is a leaky integrate-and-fire unit; every connection is
the measured synapse count times the sign of the presynaptic neuron's
transmitter. Parameters follow the whole-brain fly model of Shiu et al.
(2024, Nature 634:210), which reproduced sugar-evoked feeding and other
sensorimotor transformations from the FlyWire connectome:

    dv/dt = (x - (v - v_rest)) / tau_m        (not integrated while refractory)
    dx/dt = -x / tau_syn
    presynaptic spike -> x_post += w_syn * signed_synapse_count  (after a delay)

Sensory drive is Poisson spiking injected into identified sensory neurons.
Nothing about behaviour is programmed here: whatever reaches the descending
neurons is what the wiring diagram computes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from .connectome import Connectome


@dataclass
class LIFParams:
    dt_ms: float = 0.25
    v_rest: float = -52.0      # mV
    v_reset: float = -52.0     # mV
    v_th: float = -45.0        # mV
    tau_m: float = 20.0        # ms
    tau_syn: float = 5.0       # ms
    t_ref: float = 2.2         # ms
    delay: float = 1.8         # ms
    # Shiu et al. used 0.275 mV per FlyWire synapse. maleCNS detects ~1.77x more
    # synapses per neuron (692 vs 391 inputs), so the per-synapse weight is scaled
    # by 391/692 to keep the same total drive per neuron as the validated model.
    w_syn: float = 0.275 * 391.0 / 692.0   # = 0.155 mV per maleCNS synapse
    w_poisson: float = 0.275 * 250  # mV per Poisson input event (drives a spike)
    # Spike-frequency adaptation (absent in Shiu et al.): each spike adds
    # adapt_inc mV of hyperpolarising current that decays with tau_adapt.
    # Real fly neurons adapt; without it, recurrent loops of the dense maleCNS
    # graph (brain + nerve cord) lock into self-sustained firing after a
    # stimulus ends. Set adapt_inc = 0 to recover the original model.
    adapt_inc: float = 0.0     # mV per spike (off by default: it also weakens sugar->MN9)
    tau_adapt: float = 200.0   # ms


@numba.njit(cache=True)
def _seed(s):
    np.random.seed(s)


@numba.njit(cache=True)
def _run(nsteps, v, x, ad, ref, buf, slot, indptr, indices, data,
         in_idx, in_p, w_poi, a_m, decay, ad_inc, ad_decay, v_rest, v_th, v_reset, ref_steps, counts):
    n = v.shape[0]
    D1 = buf.shape[0]
    spk = np.empty(n, np.int64)
    for _ in range(nsteps):
        # delayed synaptic input arriving now
        b = buf[slot]
        for i in range(n):
            if b[i] != 0.0:
                x[i] += b[i]
                b[i] = 0.0
        # Poisson sensory drive
        for k in range(in_idx.shape[0]):
            if np.random.random() < in_p[k]:
                x[in_idx[k]] += w_poi
        # membrane update
        ns = 0
        for i in range(n):
            if ref[i] > 0:
                ref[i] -= 1
                v[i] = v_reset
            else:
                v[i] += a_m * (x[i] - ad[i] - (v[i] - v_rest))
                if v[i] > v_th:
                    v[i] = v_reset
                    ref[i] = ref_steps
                    counts[i] += 1
                    ad[i] += ad_inc
                    spk[ns] = i
                    ns += 1
            x[i] *= decay
            ad[i] *= ad_decay
        # schedule outgoing spikes into the delay line
        out = buf[(slot + D1 - 1) % D1]
        for s in range(ns):
            j = spk[s]
            for e in range(indptr[j], indptr[j + 1]):
                out[indices[e]] += data[e]
        slot = (slot + 1) % D1
    return slot


class Brain:
    def __init__(self, conn: Connectome, params: LIFParams | None = None, seed: int = 0):
        self.conn = conn
        self.p = p = params or LIFParams()
        W = conn.W.tocsc()
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int64)
        self.data = (W.data * p.w_syn).astype(np.float64)
        self.n = conn.n
        self.delay_steps = max(1, int(round(p.delay / p.dt_ms)))
        self.ref_steps = int(round(p.t_ref / p.dt_ms))
        self.a_m = p.dt_ms / p.tau_m
        self.decay = float(np.exp(-p.dt_ms / p.tau_syn))
        self.ad_decay = float(np.exp(-p.dt_ms / p.tau_adapt))
        self.rate = np.zeros(self.n)       # Poisson input rate (Hz) per neuron
        _seed(seed)
        self.reset()

    def reset(self):
        self.v = np.full(self.n, self.p.v_rest)
        self.x = np.zeros(self.n)
        self.ad = np.zeros(self.n)
        self.ref = np.zeros(self.n, np.int64)
        self.buf = np.zeros((self.delay_steps + 1, self.n))
        self.slot = 0
        self.t_ms = 0.0

    def set_rates(self, idx, hz):
        self.rate[np.asarray(idx)] = hz

    def clear_inputs(self):
        self.rate[:] = 0.0

    def run(self, ms: float) -> np.ndarray:
        """Advance the network by ``ms`` milliseconds; returns spike counts per neuron."""
        nsteps = int(round(ms / self.p.dt_ms))
        counts = np.zeros(self.n, np.int64)
        in_idx = np.nonzero(self.rate > 0)[0].astype(np.int64)
        in_p = self.rate[in_idx] * self.p.dt_ms * 1e-3
        self.slot = _run(nsteps, self.v, self.x, self.ad, self.ref, self.buf, self.slot,
                         self.indptr, self.indices, self.data, in_idx, in_p, self.p.w_poisson,
                         self.a_m, self.decay, self.p.adapt_inc, self.ad_decay,
                         self.p.v_rest, self.p.v_th, self.p.v_reset, self.ref_steps, counts)
        self.t_ms += nsteps * self.p.dt_ms
        return counts
