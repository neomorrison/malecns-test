"""Training curves as small multiples (one measure per panel, shared x axis).

    python scripts/plot_training.py runs/loco/train_log.jsonl results/training_curves.png
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"


def smooth(y, k=25):
    if len(y) < k:
        return np.asarray(y)
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, np.nan), c])


def main(log_path, out_path):
    recs = [json.loads(l) for l in open(log_path)]
    its = np.array([r["it"] for r in recs])
    order = np.argsort(its, kind="stable")
    # keep the last record per iteration (the run was resumed once)
    seen, idx = set(), []
    for i in order[::-1]:
        if its[i] not in seen:
            seen.add(its[i])
            idx.append(i)
    recs = [recs[i] for i in sorted(idx, key=lambda i: its[i])]
    hours = np.array([r["samples"] for r in recs], float)
    # samples restart at 0 after the resume; make them cumulative
    for i in range(1, len(hours)):
        if hours[i] < hours[i - 1]:
            hours[i:] += hours[i - 1]
    hours = hours * 0.02 / 3600.0
    ep_len = np.array([r["ep_len"] for r in recs]) * 0.02
    it = np.array([r["it"] for r in recs])
    lin = np.array([r["terms"]["lin_vel"] for r in recs]) / 1.5
    yaw_w = np.where(it <= 2100, 0.7, 1.0)
    yaw = np.array([r["terms"]["yaw_rate"] for r in recs]) / yaw_w
    vmax = np.array([r["vx_max"] for r in recs])
    events = []
    for a, b in zip(recs[:-1], recs[1:]):
        if b["pushes"] and not a["pushes"]:
            events.append((b, "pushes + jumps\nswitched on"))
    resume_h = hours[np.searchsorted(it, 2101)] if (it > 2100).any() else None

    panels = [("Time before falling (s, max 20)", smooth(ep_len), "{:.1f} s"),
              ("Follows speed command (0-1)", smooth(lin), "{:.2f}"),
              ("Follows turn command (0-1)", smooth(yaw), "{:.2f}"),
              ("Top speed asked of it (m/s)", vmax, "{:.1f} m/s")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, facecolor=SURFACE)
    for ax, (title, y, fmt) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        ax.plot(hours, y, color=SERIES, lw=2, solid_capstyle="round", solid_joinstyle="round")
        last = np.where(np.isfinite(y))[0][-1]
        ax.annotate(fmt.format(y[last]), (hours[last], y[last]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
        ax.grid(axis="y", color=GRID, lw=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.set_xlim(0, hours[-1] * 1.12)
        if resume_h is not None:
            ax.axvline(resume_h, color=INK2, lw=1, alpha=0.5)
        for rec, label in events:
            h = hours[[r["it"] for r in recs].index(rec["it"])]
            ax.axvline(h, color=INK2, lw=1, alpha=0.5)
    axes[1, 1].set_ylim(0, 3.8)
    if resume_h is not None:
        axes[1, 0].text(resume_h, axes[1, 0].get_ylim()[1], " turning reward fixed", color=INK2, fontsize=8, va="top")
    for rec, label in events:
        h = hours[[r["it"] for r in recs].index(rec["it"])]
        axes[1, 1].text(h, 3.7, " " + label, color=INK2, fontsize=8, va="top")
    for ax in axes[1]:
        ax.set_xlabel("simulated practice (hours of experience across 256 parallel bodies)", color=INK2)
    fig.suptitle(f"The humanoid learning to walk  -  iteration {recs[-1]['it']:,}, "
                 f"{recs[-1]['samples'] / 1e6 + (sum(r['samples'] for r in recs if r['it'] == 2100) / 1e6):.0f} M steps",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
