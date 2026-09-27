"""Probe the maleCNS spiking model: what does each sense make the brain do?

    python scripts/probe_brain.py            # writes results/brain_probe.{json,md}

For each sensory channel (sugar taste, visual target left/right, looming
left/right, food odour, the hunger drive, and no input) we drive the
identified neurons with Poisson spikes and record the firing of every
descending neuron. This is the evidence behind the motor decoder: which
command neurons each sense actually reaches through the wiring diagram.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from malecns_body import connectome as C  # noqa: E402
from malecns_body.brain import Brain, LIFParams  # noqa: E402
from malecns_body.circuits import build_interface, group_rates  # noqa: E402


def main(ms=1000.0, seeds=(0, 1, 2), rate=100.0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    conn = C.load()
    iface = build_interface(conn)
    print(iface.summary())
    brain = Brain(conn, LIFParams(), seed=0)
    a = conn.ann
    dn = np.nonzero((a["superclass"] == "descending_neuron").to_numpy())[0]
    fwd = np.concatenate(iface.readout["FORWARD"])
    conds = {
        "none": [],
        "sugar": [iface.sugar],
        "target_left": [iface.pursuit_l],
        "target_right": [iface.pursuit_r],
        "loom_left": [iface.loom_l],
        "loom_right": [iface.loom_r],
        "hunger_drive": [fwd],
        "odor_left": [iface.odor_l],
        "odor_right": [iface.odor_r],
    }
    results = {}
    for name, groups in conds.items():
        rates = []
        grp = []
        t0 = time.time()
        for s in seeds:
            brain.reset()
            brain.clear_inputs()
            for g in groups:
                brain.set_rates(g, rate)
            from malecns_body.brain import _seed
            _seed(s)
            c = brain.run(ms)
            rates.append(c * 1000.0 / ms)
            grp.append(group_rates(iface, c, ms))
        r = np.mean(rates, 0)
        g = {k: [float(np.mean([x[k][0] for x in grp])), float(np.mean([x[k][1] for x in grp]))] for k in grp[0]}
        top = dn[np.argsort(-r[dn])[:15]]
        results[name] = dict(
            groups=g,
            total_rate=float(r.mean()),
            active_neurons=int((r > 1).sum()),
            top_descending=[dict(instance=str(a.instance.values[i]), type=str(a.type.values[i]),
                                 rate=round(float(r[i]), 1)) for i in top if r[i] > 0],
        )
        print(f"\n[{name}] ({time.time()-t0:.1f}s)  mean rate {r.mean():.2f} Hz, active {results[name]['active_neurons']}")
        for k, (l, rr) in g.items():
            print(f"   {k:9s} L {l:6.1f}  R {rr:6.1f} Hz")
        print("   top DNs:", ", ".join(f"{d['instance']}={d['rate']}" for d in results[name]["top_descending"][:8]))
    with open(os.path.join(out_dir, "brain_probe.json"), "w") as f:
        json.dump(dict(params=LIFParams().__dict__, stim_rate_hz=rate, window_ms=ms, seeds=list(seeds),
                       n_neurons=conn.n, interface=iface.summary(), results=results), f, indent=1)
    with open(os.path.join(out_dir, "brain_probe.md"), "w") as f:
        f.write("# maleCNS spiking-model probe\n\n")
        f.write(f"Brain model: {conn.n:,} neurons. Each channel driven at {rate:.0f} Hz Poisson for "
                f"{ms:.0f} ms, mean of {len(seeds)} seeds.\n")
        f.write("Rates in Hz: mean over the left / right cells of each readout group.\n\n```\n"
                + iface.summary() + "\n```\n\n")
        keys = list(next(iter(results.values()))["groups"])
        f.write("| stimulus | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n")
        for name, res in results.items():
            f.write(f"| {name} | " + " | ".join(f"{res['groups'][k][0]:.0f} / {res['groups'][k][1]:.0f}" for k in keys) + " |\n")
        f.write("\n## Most active descending neurons per stimulus\n\n")
        for name, res in results.items():
            f.write(f"- **{name}**: " + ", ".join(f"{d['instance']} ({d['rate']:.0f})" for d in res["top_descending"][:10]) + "\n")
    return results


if __name__ == "__main__":
    main()
