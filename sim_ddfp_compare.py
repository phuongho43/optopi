"""Re-run the Figure 3G OptoPI sparse-decoder comparison.

Three questions, one script:
  A. does the fitted model still match on the NEW decoder dataset (7--decoder-new),
     the one panels 2H/2I are being swapped to?
  B. what does the model do BETWEEN camera frames? (model runs at 0.1 s, data at 5.1 s)
  C. what would the decoder do under the Figures 4-6 working inputs (25% duty sparse,
     continuous dense) instead of the microscope-constrained 10%/80%?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from optopi.fit_model import prep_tu_tye_data
from optopi.model import sim_sparser
from optopi.sim_model import prep_sparser_params

REPO = Path("/home/phuong/projects/csc-revisions-2026")
KLOV = REPO / "data/2--optopi/0--LOVfast/fit_params.json"
KILID = REPO / "data/2--optopi/3--iLIDslow/fit_params.json"
OUT = REPO / "data/2--optopi/7--ddfp-resolution"
OUT.mkdir(parents=True, exist_ok=True)


def run(class_dp, tmax=None):
    tu, ty = prep_tu_tye_data(class_dp)
    if tmax:
        tu, ty = tu[tu.t < tmax], ty[ty.t < tmax]
    yy = ty.y.to_numpy() - ty.y.to_numpy().min()
    ty = ty.assign(y=yy)
    y0 = [0, 0, 0, 0, yy.max(), yy.max(), 0, 0, yy[0]]
    kk = prep_sparser_params(KLOV, KILID)
    tm, ym = sim_sparser(tu.t.to_numpy(), y0, tu.u.to_numpy(), kk)
    return tu, ty, pd.DataFrame({"t": tm, "y": ym[-1]})


def ideal(tmax, sparse_win, dense_win, sparse_period=4.0, width=1.0, y_scale=1.0, y_init=0.0):
    """Figures 4-6 working inputs: 25% duty sparse, continuous dense."""
    tt = np.round(np.arange(0, tmax + 0.1, 0.1), 1)
    uu = np.zeros_like(tt)
    for s in np.arange(sparse_win[0], sparse_win[1], sparse_period):
        uu[(tt >= s) & (tt < s + width)] = 1.0
    uu[(tt >= dense_win[0]) & (tt < dense_win[1])] = 1.0
    y0 = [0, 0, 0, 0, y_scale, y_scale, 0, 0, y_init]
    kk = prep_sparser_params(KLOV, KILID)
    tm, ym = sim_sparser(tt, y0, uu, kk)
    return pd.DataFrame({"t": tt, "u": uu}), pd.DataFrame({"t": tm, "y": ym[-1]})


if __name__ == "__main__":
    for name, dp, tmax in [
        ("old_5decoder", REPO / "data/1--biosensor/5--decoder/0--sparse-ddFP", 155),
        ("new_7decoder", REPO / "data/1--biosensor/7--decoder-new/2--sparse-ddFP", None),
    ]:
        tu, ty, tm = run(dp, tmax)
        tu.to_csv(OUT / f"{name}_input.csv", index=False)
        ty.to_csv(OUT / f"{name}_data.csv", index=False)
        tm.to_csv(OUT / f"{name}_model.csv", index=False)
        dt_model = np.median(np.diff(tm.t))
        dt_data = np.median(np.diff(ty.t))
        print(f"{name}: model dt={dt_model:.2f}s ({len(tm)} pts)  data dt={dt_data:.2f}s ({len(ty)} pts)")
        # goodness of fit at the sampled points
        yi = np.interp(ty.t, tm.t, tm.y)
        ss = 1 - np.sum((ty.y - yi) ** 2) / np.sum((ty.y - ty.y.mean()) ** 2)
        print(f"    R^2 at sampled points = {ss:.3f}   data range [{ty.y.min():.3f}, {ty.y.max():.3f}]" f"   model range [{tm.y.min():.3f}, {tm.y.max():.3f}]")
    # C: idealized working inputs, same window layout AND same initial conditions /
    # scale as the new-dataset run, so the two curves are directly comparable.
    _, ty_new, _ = run(REPO / "data/1--biosensor/7--decoder-new/2--sparse-ddFP")
    yy_new = ty_new.y.to_numpy()
    tu_i, tm_i = ideal(320, (60, 186), (186, 320), y_scale=yy_new.max(), y_init=yy_new[0])
    tu_i.to_csv(OUT / "ideal_input.csv", index=False)
    tm_i.to_csv(OUT / "ideal_model.csv", index=False)
    print("\n  regime means, same y0/scale for both (model, 0.1 s grid):")
    _, _, tm_new = run(REPO / "data/1--biosensor/7--decoder-new/2--sparse-ddFP")
    for lbl, a, b in [("None", 0, 60), ("Sparse", 60, 186), ("Dense", 186, 320)]:
        m = tm_i[(tm_i.t >= a) & (tm_i.t < b)].y
        d = tm_new[(tm_new.t >= a) & (tm_new.t < b)].y
        print(f"    {lbl:7} measured-input model={d.mean():+.4f}   working-input model={m.mean():+.4f}" f"   (working peak {m.max():+.4f})")


def regenerate_3g(out_name="fig_3g_sparse_model_NEW.png"):
    """Rebuild the Figure 3G panel against 7--decoder-new, in the house style.

    Does not reuse optopi.sim_model.plot_model_fit: that draws one axvspan per
    0.1 s sample where u > 0, which is ~2000 patches on this 320 s recording and
    blows the recursion limit during save. Contiguous on-runs are merged instead.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from optopi.style import STYLE

    dp = REPO / "data/1--biosensor/7--decoder-new/2--sparse-ddFP"
    tu, ty, tm = run(dp)

    # merge contiguous u > 0 samples into spans
    t, u = tu.t.to_numpy(), tu.u.to_numpy()
    on = u > 0
    spans, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            spans.append((t[i], t[j] + (t[1] - t[0])))
            i = j + 1
        else:
            i += 1

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(24, 16))
        for a, b in spans:
            ax.axvspan(a, b, color="#648FFF", lw=0, alpha=0.35, zorder=1)
        ax.plot(tm.t, tm.y, color="#2ECC71", lw=STYLE["lines.linewidth"], zorder=2.2, label="Model")
        ax.plot(ty.t, ty.y, color="#34495E", lw=0, marker="o", markersize=STYLE["lines.markersize"], markeredgewidth=2, markeredgecolor="#212121", zorder=2.3, label="Data")
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", lw=STYLE["lines.linewidth"]),
            mpl.lines.Line2D([], [], color="#34495E", marker="o", lw=0),
            mpl.lines.Line2D([], [], color="#2ECC71", lw=STYLE["lines.linewidth"]),
        ]
        ax.legend(handles, ["Input", "Data", "Model"], loc="upper left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ddFP Response ($\\Delta$F/F$_0$)")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig_fp = REPO / "figures/not-used" / out_name
        fig.savefig(fig_fp, dpi=100)
        plt.close("all")

    yi = np.interp(ty.t, tm.t, tm.y)
    r2 = 1 - np.sum((ty.y - yi) ** 2) / np.sum((ty.y - ty.y.mean()) ** 2)
    print(f"wrote {fig_fp}")
    print(f"  R^2 = {r2:.3f} | {len(ty)} data pts (dt {np.median(np.diff(ty.t)):.1f} s)" f" | {len(tm)} model pts (dt {np.median(np.diff(tm.t)):.2f} s) | {len(spans)} pulses")
    return fig_fp
