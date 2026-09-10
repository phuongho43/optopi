"""Re-plot the S3-7 OptoPI screen from saved CSVs (no re-simulation), in optopi house style.

Headline: kd1 (LOV dark-reversion) tunes the sparse-decoder response frequency.
Supporting: dense-input crosstalk invariant to the LOV repression module; kb1 tunes amplitude.

    uv run python plot_crosstalk_screen.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.path as _mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_mpath.Path.__deepcopy__ = lambda self, memo=None: _mpath.Path(self.vertices.copy(), self.codes.copy() if self.codes is not None else None)

D = Path("/home/phuong/projects/csc-revisions-2026/data/2--optopi/5--crosstalk-screen")
curves = pd.read_csv(D / "crosstalk_curves.csv")
summ = pd.read_csv(D / "crosstalk_screen.csv")
MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
INK, PURPLE, ORANGE, NAVY, GREEN = "#212121", "#8069EC", "#EA822C", "#34495E", "#2ECC71"

HOUSE = {
    "font.size": 32,
    "axes.titlesize": 38,
    "axes.labelsize": 40,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 24,
    "lines.linewidth": 7,
    "lines.markersize": 16,
    "lines.markeredgecolor": INK,
    "lines.markeredgewidth": 2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelweight": 600,
    "axes.edgecolor": INK,
    "axes.linewidth": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 12,
    "ytick.major.size": 12,
    "xtick.major.width": 4,
    "ytick.major.width": 4,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
}


def curve(param, mult):
    s = curves[(curves.param == param) & (np.isclose(curves.mult, mult))].sort_values("freq")
    return s.freq.to_numpy(), s.output.to_numpy()


def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=52, fontweight=700, color=INK, va="bottom")


with mpl.rc_context(HOUSE):
    fig, ax = plt.subplots(1, 3, figsize=(34, 10))
    oranges = plt.get_cmap("Oranges")

    # A: kd1 sweep FM curves
    fd, od = curve("dense_ref", 1.0)
    ax[0].plot(fd, od, color=PURPLE, ls="--", lw=6, label="dense (ref)")
    for i, m in enumerate(MULTS):
        f, o = curve("kd1", m)
        ax[0].plot(f, o, color=oranges(0.35 + 0.6 * i / (len(MULTS) - 1)), label=f"kd1×{m}" + (" (fit)" if m == 1.0 else ""))
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Input frequency (Hz)")
    ax[0].set_ylabel("Mean sparse output (AU)")
    ax[0].set_title("kd1 tunes response frequency", pad=14)
    ax[0].legend(loc="upper left", frameon=False)
    panel_label(ax[0], "A")

    # B: peak-frequency & selectivity vs kd1
    k = summ[summ.param == "kd1"].sort_values("mult")
    ax[1].plot(k["mult"], k["peak_freq"], marker="o", color=ORANGE, label="peak frequency")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("kd1 × fold")
    ax[1].set_ylabel("Sparse peak freq (Hz)", color=ORANGE)
    ax[1].tick_params(axis="y", labelcolor=ORANGE)
    ax[1].set_title("Reversion rate sets the peak", pad=14)
    b2 = ax[1].twinx()
    b2.spines["top"].set_visible(False)
    b2.plot(k["mult"], k["selectivity"], marker="s", color=INK, ls=":", lw=6, label="selectivity")
    b2.set_ylabel("selectivity (peak / dense)", color=INK)
    b2.tick_params(axis="y", labelcolor=INK, labelsize=30)
    panel_label(ax[1], "B")

    # C: crosstalk@1Hz invariance
    for p, c in zip(["kb1", "kl1", "kd1"], [NAVY, GREEN, ORANGE], strict=False):
        s = summ[summ.param == p].sort_values("mult")
        ax[2].plot(s["mult"], s["crosstalk_1Hz"], marker="o", color=c, label=p)
    pk = float(summ["peak"].max())
    ax[2].axhline(pk, color="#9aa0a6", ls="--", lw=5, label=f"peak (~{pk:.2f})")
    ax[2].set_xscale("log")
    ax[2].set_ylim(0, pk * 1.12)
    ax[2].set_xlabel("parameter × fold")
    ax[2].set_ylabel("crosstalk @1 Hz (AU)")
    ax[2].set_title("Dense crosstalk ≈ 0, invariant", pad=14)
    ax[2].legend(loc="center left", frameon=False)
    panel_label(ax[2], "C")

    fig.tight_layout(w_pad=4)
    out = D / "crosstalk_screen_final.png"
    fig.savefig(out)
    plt.close(fig)
print("saved", out)
