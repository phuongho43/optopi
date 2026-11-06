"""S3-6 / R3.1: can kinetic tuning of the EXISTING LOV/iLID sparse motif yield a 3rd,
separable frequency (temporal-pattern) channel?

Channel 1 = dense decoder (iLIDfast, high-pass). Channels 2/3 = sparse-motif variants whose
reversion kinetics (kd1 = LOV dark-reversion, kd2 = iLIDslow reversion) are scaled to move the
band-pass peak. We (a) map peak-frequency reachability over the (kd1, kd2) grid, (b) pick a
3-decoder bank with maximally separated peaks, and (c) score separability with a confusion
matrix: at each channel's peak input frequency, is only that channel high?

    uv run python screen_3decoder.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.path as _mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optopi.fit_model import prep_tu_tye_data
from optopi.model import sim_lid, sim_sparser
from optopi.sim_model import prep_sparser_params

_mpath.Path.__deepcopy__ = lambda self, memo=None: _mpath.Path(self.vertices.copy(), self.codes.copy() if self.codes is not None else None)

DATA = Path("/home/phuong/projects/csc-revisions-2026/data")
OPTO = DATA / "2--optopi"
BIO = DATA / "1--biosensor"
OUT = OPTO / "6--three-decoder-search"
OUT.mkdir(parents=True, exist_ok=True)

# 121 dropped Nov 2026: uu[121:121:121] is an EMPTY slice -- no light delivered.
PERIODS = sorted({*range(1, 21), 22, 24, 26, 28, 30, 33, 36, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110})
TU = np.arange(0, 121, 1)
KD1S = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # LOV dark-reversion folds
KD2S = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # iLIDslow reversion folds (accumulation timescale)


def sparse_y0():
    _, ty = prep_tu_tye_data(BIO / "5--decoder" / "0--sparse-ddFP")
    yy = ty["y"].to_numpy()
    yy = yy - np.min(yy)
    return [0, 0, 0, 0, float(np.max(yy)), float(np.max(yy)), 0, 0, float(yy[0])]


def dense_y0():
    _, ty = prep_tu_tye_data(BIO / "3--iLID" / "0--I427V")
    yy = ty["y"].to_numpy()
    return [float(np.max(yy)), 0, float(np.max(yy)), 0]


def fm_curve(sim, y0, kk):
    freqs, outs = [], []
    for period in PERIODS:
        uu = np.zeros_like(TU)
        uu[period:121:period] = 1
        _, ym = sim(TU, y0, uu, kk)
        freqs.append(1.0 / period)
        outs.append(float(np.mean(ym[-1])))
    f = np.array(freqs)
    o = np.array(outs)
    order = np.argsort(f)
    return f[order], o[order]


def main():
    y0s, y0d = sparse_y0(), dense_y0()
    with open(OPTO / "2--iLIDfast" / "fit_params.json") as fh:
        k_ilidfast = json.load(fh)
    base = prep_sparser_params(OPTO / "0--LOVfast" / "fit_params.json", OPTO / "3--iLIDslow" / "fit_params.json")

    fd, od = fm_curve(sim_lid, y0d, k_ilidfast)  # dense channel (high-pass)
    dense_peak_f = float(fd[np.argmax(od)])

    # ---- (a) peak-frequency reachability over (kd1, kd2) ----
    grid, variants = [], {}
    for m1 in KD1S:
        for m2 in KD2S:
            kk = dict(base)
            kk["kd1"] *= m1
            kk["kd2"] *= m2
            f, o = fm_curve(sim_sparser, y0s, kk)
            pf = float(f[np.argmax(o)])
            pk = float(np.max(o))
            variants[(m1, m2)] = (f, o, pf, pk)
            grid.append({"kd1_mult": m1, "kd2_mult": m2, "peak_freq": pf, "peak": pk})
            print(f"  kd1x{m1} kd2x{m2}: peak={pk:.3f} @ {pf:.4f} Hz")
    gdf = pd.DataFrame(grid)
    gdf.to_csv(OUT / "reachability.csv", index=False)
    print(f"\nDense peak @ {dense_peak_f:.3f} Hz")
    print(f"Sparse peak-freq reachable range: {gdf.peak_freq.min():.4f} - {gdf.peak_freq.max():.4f} Hz")

    # ---- (b) pick a 3-decoder bank: dense + two sparse variants with peaks maximally
    # separated from each other and from dense (require decent peak amplitude) ----
    ok = gdf[gdf.peak > 0.15]  # usable amplitude
    slow = ok.loc[ok.peak_freq.idxmin()]  # lowest-freq sparse channel
    # mid channel: variant whose peak freq best splits (log) the gap between slow and dense
    target_mid = np.sqrt(slow.peak_freq * dense_peak_f)
    ok = ok.assign(dist=(np.log(ok.peak_freq) - np.log(target_mid)).abs())
    mid = ok.loc[ok.dist.idxmin()]
    bank = {
        "dense": (fd, od, dense_peak_f),
        f"mid (kd1x{mid.kd1_mult},kd2x{mid.kd2_mult})": (*variants[(mid.kd1_mult, mid.kd2_mult)][:2], mid.peak_freq),
        f"slow (kd1x{slow.kd1_mult},kd2x{slow.kd2_mult})": (*variants[(slow.kd1_mult, slow.kd2_mult)][:2], slow.peak_freq),
    }
    names = list(bank)
    target_freqs = [bank[n][2] for n in names]

    # ---- (c) confusion matrix: channel response at each target freq, normalized to own peak ----
    M = np.zeros((3, 3))
    for i, n in enumerate(names):
        f, o, _ = bank[n]
        onorm = o / np.max(o)
        for j, tf in enumerate(target_freqs):
            M[i, j] = onorm[np.argmin(np.abs(f - tf))]
    # separability: for each target column, correct channel minus best competitor
    margins = [M[j, j] - max(M[i, j] for i in range(3) if i != j) for j in range(3)]
    print("\nBank channels:", names)
    print("Target freqs (Hz):", [round(t, 4) for t in target_freqs])
    print("Confusion matrix (row=channel, col=target freq; 1=own peak):\n", np.round(M, 2))
    print("Per-channel separation margins:", [round(m, 2) for m in margins], "-> min margin =", round(min(margins), 2))
    pd.DataFrame(M, index=names, columns=[f"{t:.3f}Hz" for t in target_freqs]).to_csv(OUT / "confusion.csv")
    # save bank curves (re-plottable without re-simulating)
    bank_rows = []
    for n in names:
        f, o, pf = bank[n]
        bank_rows += [{"channel": n.split(" ")[0], "label": n, "peak_freq": pf, "freq": ff, "output": oo} for ff, oo in zip(f, o, strict=False)]
    pd.DataFrame(bank_rows).to_csv(OUT / "bank_curves.csv", index=False)

    # ---- plots (optopi house style) ----
    INK = "#212121"
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

    def plabel(ax, s):
        ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=52, fontweight=700, color=INK, va="bottom")

    with mpl.rc_context(HOUSE):
        fig, ax = plt.subplots(1, 3, figsize=(34, 10))
        # A: reachability heatmap
        piv = gdf.pivot(index="kd2_mult", columns="kd1_mult", values="peak_freq")
        im = ax[0].imshow(piv.values, origin="lower", aspect="auto", cmap="viridis", extent=[0, len(KD1S), 0, len(KD2S)])
        ax[0].set_xticks(np.arange(len(KD1S)) + 0.5)
        ax[0].set_xticklabels(KD1S)
        ax[0].set_yticks(np.arange(len(KD2S)) + 0.5)
        ax[0].set_yticklabels(KD2S)
        ax[0].set_xlabel("kd1 × fold")
        ax[0].set_ylabel("kd2 × fold")
        ax[0].set_title("Sparse peak freq (Hz)", pad=14)
        cb = fig.colorbar(im, ax=ax[0])
        cb.ax.tick_params(labelsize=26)
        plabel(ax[0], "A")
        # B: 3-channel bank
        palette = ["#8069EC", "#EA822C", "#34495E"]
        for k, n in enumerate(names):
            f, o, pf = bank[n]
            ax[1].plot(f, o / np.max(o), color=palette[k], label=n.split(" ")[0])
            ax[1].axvline(pf, color=palette[k], ls=":", lw=4)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Input frequency (Hz)")
        ax[1].set_ylabel("Normalized output")
        ax[1].set_title("Candidate 3-decoder bank", pad=14)
        ax[1].legend(loc="upper center", frameon=False)
        plabel(ax[1], "B")
        # C: confusion matrix
        im2 = ax[2].imshow(M, cmap="magma", vmin=0, vmax=1)
        ax[2].set_xticks(range(3))
        ax[2].set_xticklabels([f"{t:.3f}\nHz" for t in target_freqs])
        ax[2].set_yticks(range(3))
        ax[2].set_yticklabels([n.split(" ")[0] for n in names])
        for i in range(3):
            for j in range(3):
                ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=34, color="white" if M[i, j] < 0.6 else INK)
        ax[2].set_xlabel("target input frequency")
        ax[2].set_ylabel("decoder channel")
        ax[2].set_title(f"Separability (min margin {min(margins):.2f})", pad=14)
        cb2 = fig.colorbar(im2, ax=ax[2])
        cb2.ax.tick_params(labelsize=26)
        plabel(ax[2], "C")
        fig.tight_layout(w_pad=4)
        fig.savefig(OUT / "three_decoder_search.png")
        plt.close(fig)
    print("\nsaved", OUT / "three_decoder_search.png")


if __name__ == "__main__":
    main()
