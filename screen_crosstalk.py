"""S3-7 (R3.2) OptoPI parameter screen: sparse-decoder dense-input crosstalk & freq-response shift.

Sweeps the LOVfast repression module (kb1 primary; kl1, kd1) of the sparse-decoder ODE
(`optopi.model.sim_sparser`) and, for each value, recomputes the mean-output-vs-input-frequency
curve. Reports:
  - crosstalk  = sparse mean output at dense input (1 Hz, period = 1)
  - selectivity = peak output / crosstalk (higher = cleaner frequency discrimination)
  - peak-freq   = input frequency at which sparse output is maximal (freq-response shift)

Baseline params are the fitted LOVfast + iLIDslow rates; y0 is taken from the sparse-ddFP
biosensor data exactly as in optopi.sim_model.prep_and_sim_fm_ave (faithful baseline).

    uv run python screen_crosstalk.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.path as _mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib<3.9 + Python 3.14: Path.__deepcopy__ recurses infinitely in legend building. Patch it.
_mpath.Path.__deepcopy__ = lambda self, memo=None: _mpath.Path(self.vertices.copy(), self.codes.copy() if self.codes is not None else None)

from optopi.fit_model import prep_tu_tye_data  # noqa: E402  (must follow the Path.__deepcopy__ patch above)
from optopi.model import sim_lid, sim_sparser, sparser_dark_equilibrium, sparser_signal  # noqa: E402  (must follow the Path.__deepcopy__ patch above)
from optopi.sim_model import prep_sparser_params  # noqa: E402  (must follow the Path.__deepcopy__ patch above)

DATA = Path("/home/phuong/projects/csc-revisions-2026/data")
OPTO = DATA / "2--optopi"
BIO = DATA / "1--biosensor"
OUT = OPTO / "5--crosstalk-screen"
OUT.mkdir(parents=True, exist_ok=True)

# 121 dropped Nov 2026: uu[121:121:121] is an EMPTY slice, so that point delivered no
# light at all -- a dark control plotted as a 0.00826 Hz stimulus. See optopi/model.py.
PERIODS = sorted({*range(1, 21), 22, 24, 26, 28, 30, 33, 36, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110})  # fine grid (pub quality)
TU = np.arange(0, 121, 1)
MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]  # log-spaced fold-range around fitted value
TARGETS = ["kb1", "kl1", "kd1"]  # LOV repression module (kb1 primary per P.H.)
DENSE_FREQ = 1.0  # period = 1 -> dense/continuous input


def sparse_y0(kk):
    """The sparse decoder's dark equilibrium.

    Was the data's own offset handed in as AiBaC(0). The model now has a dark-state binding
    reaction, so the resting state is a property of the parameters rather than of a
    recording. It depends only on kb1, kb2d and kd2d, so one call serves the whole reversion
    sweep -- kd1 and kd2 enter through terms that are zero in the dark.
    """
    return sparser_dark_equilibrium(kk)


def dense_y0():
    _, ty = prep_tu_tye_data(BIO / "3--iLID" / "0--I427V")
    yy = ty["y"].to_numpy()
    return [float(np.max(yy)), 0, float(np.max(yy)), 0]


def fm_curve(sim, y0, kk):
    """Mean output (last state) vs input frequency."""
    freqs, outs = [], []
    for period in PERIODS:
        uu = np.zeros_like(TU)
        uu[period:121:period] = 1
        _, ym = sim(TU, y0, uu, kk)
        # Read the sparse decoder's signal through sparser_signal, NOT ym[-1]: it is now the
        # sum of the lit and dark ternary complexes, and the positional convention would
        # silently return the dark one alone while still producing a plausible curve. The
        # resting level is subtracted because the confusion matrix normalises each channel
        # to its own peak, so a constant baseline would not divide out.
        y = sparser_signal(ym) if sim is sim_sparser else ym[-1]
        freqs.append(1.0 / period)
        outs.append(float(np.mean(y - y[0])))
    return np.array(freqs), np.array(outs)


def metrics(freqs, outs):
    order = np.argsort(freqs)
    f, o = freqs[order], outs[order]
    crosstalk = float(o[np.argmin(np.abs(f - DENSE_FREQ))])  # output at 1 Hz
    ipk = int(np.argmax(o))
    peak, peak_freq = float(o[ipk]), float(f[ipk])
    selectivity = float(peak / crosstalk) if crosstalk > 0 else np.inf
    return crosstalk, peak, peak_freq, selectivity


def main():
    y0d = dense_y0()
    with open(OPTO / "2--iLIDfast" / "fit_params.json") as fh:
        k_ilidfast = json.load(fh)
    kk_base = prep_sparser_params(OPTO / "0--LOVfast" / "fit_params.json", OPTO / "3--iLIDslow" / "fit_params.json")
    # The rest state now follows from the parameters (dark-state binding), not from
    # the recording. Unaffected by the rates this screen scales.
    y0s = sparse_y0(kk_base)
    print("Sparse-decoder base params:", {k: round(v, 4) for k, v in kk_base.items()})

    # dense decoder reference curve
    fd, od = fm_curve(sim_lid, y0d, k_ilidfast)
    # sparse baseline
    fb, ob = fm_curve(sim_sparser, y0s, kk_base)
    ct, pk, pf, sel = metrics(fb, ob)
    print(f"\nBASELINE sparse: crosstalk(@1Hz)={ct:.4f}  peak={pk:.4f} @ {pf:.3f} Hz  selectivity={sel:.1f}")

    # sweep
    rows, curves = [], {}
    for tgt in TARGETS:
        for m in MULTS:
            kk = dict(kk_base)
            kk[tgt] = kk_base[tgt] * m
            f, o = fm_curve(sim_sparser, y0s, kk)
            ct, pk, pf, sel = metrics(f, o)
            curves[(tgt, m)] = (f, o)
            rows.append({"param": tgt, "mult": m, "value": kk[tgt], "crosstalk_1Hz": ct, "peak": pk, "peak_freq": pf, "selectivity": sel})
            print(f"  {tgt} x{m:<4}: crosstalk={ct:.4f}  peak={pk:.4f}@{pf:.3f}Hz  sel={sel:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crosstalk_screen.csv", index=False)

    # save all FM curves (for re-plotting without re-simulating)
    crows = [{"param": "baseline", "mult": 1.0, "freq": f, "output": o} for f, o in zip(fb, ob, strict=False)]
    crows += [{"param": "dense_ref", "mult": 1.0, "freq": f, "output": o} for f, o in zip(fd, od, strict=False)]
    for (tgt, m), (f, o) in curves.items():
        crows += [{"param": tgt, "mult": m, "freq": ff, "output": oo} for ff, oo in zip(f, o, strict=False)]
    pd.DataFrame(crows).to_csv(OUT / "crosstalk_curves.csv", index=False)

    # ---- plot: one panel per swept param (FM curves by multiplier) + summary ----
    cmap = plt.get_cmap("viridis")
    with mpl.rc_context({"font.size": 22, "lines.linewidth": 3, "figure.dpi": 150}):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        for ax, tgt in zip(axes.flat[:3], TARGETS, strict=False):
            ax.plot(np.sort(fd), od[np.argsort(fd)], color="#8069EC", ls="--", lw=2.5, label="dense decoder (ref)")
            for i, m in enumerate(MULTS):
                f, o = curves[(tgt, m)]
                order = np.argsort(f)
                ax.plot(f[order], o[order], color=cmap(i / (len(MULTS) - 1)), label=f"{tgt}×{m}" + (" (fit)" if m == 1.0 else ""))
            ax.axvline(DENSE_FREQ, color="#888", ls=":", lw=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("Input frequency (Hz)")
            ax.set_ylabel("Mean sparse output (AU)")
            ax.set_title(f"Sweep {tgt}")
            ax.legend(fontsize=13, loc="upper left")
        # summary: crosstalk & selectivity vs multiplier
        axs = axes.flat[3]
        for tgt in TARGETS:
            sub = df[df.param == tgt].sort_values("mult")
            axs.plot(sub.mult, sub.crosstalk_1Hz, marker="o", label=f"{tgt}: crosstalk@1Hz")
        axs.set_xscale("log")
        axs.set_xlabel("parameter × fold")
        axs.set_ylabel("crosstalk (sparse output @ 1 Hz)")
        axs.set_title("Dense-input crosstalk vs repression strength")
        axs.legend(fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / "crosstalk_screen.png")
        plt.close(fig)
    print("\nsaved:", OUT / "crosstalk_screen.png", "and crosstalk_screen.csv")


if __name__ == "__main__":
    main()
