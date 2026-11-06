"""Fit the one parameter the biosensor data cannot constrain: dark-state iLID/sspB affinity.

dF/F0 is referenced to the pre-stimulus baseline, so a constant dark complex is subtracted
away by the normalisation itself -- the iLID biosensor traces behind Figures 2D-2F carry no
information about it. It becomes visible only in the sparse decoder, where releasing LOV from
Zdk destroys the dark complex and drives the signal BELOW its dark baseline. That excursion
(11% of the range) is what this fits.

One free parameter, expressed as a dissociation constant in units of the protein pool, with
association taken as diffusion-limited and unchanged by the photostate. The observation model
carries no offset: y(t) = s * (signal(t) - signal_dark), with signal_dark the model's own
equilibrium, so the resting level is predicted rather than imposed.

    uv run python fit_dark_affinity.py [--dry-run]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from optopi.fit_model import prep_tu_tye_data
from optopi.model import sim_sparser, sparser_dark_equilibrium, sparser_signal

REPO = Path("/home/phuong/projects/csc-revisions-2026")
OPTO = REPO / "data/2--optopi"
RECORDING = REPO / "data/1--biosensor/5--decoder/0--sparse-ddFP"
OUT = OPTO / "7--ddfp-resolution/dark_affinity.json"
TMAX = 155.0


def load():
    tu, ty = prep_tu_tye_data(RECORDING)
    tu, ty = tu[tu.t < TMAX], ty[ty.t < TMAX]
    return (tu.t.to_numpy(), tu.u.to_numpy(), ty.t.to_numpy(), ty.y.to_numpy())


def predict(kd_over_pool, tt, uu, td, kk):
    kk = dict(kk)
    kk["kb2d"] = kk["kb2"]
    kk["kd2d"] = kd_over_pool * kk["kb2"]
    y0 = sparser_dark_equilibrium(kk)
    _, ym = sim_sparser(tt, y0, uu, kk)
    sig = sparser_signal(ym)
    d = np.interp(td, tt, sig - sig[0])
    return d, sig, ym


def score(kd_over_pool, tt, uu, td, raw, kk):
    d, sig, _ = predict(kd_over_pool, tt, uu, td, kk)
    denom = float(np.dot(d, d))
    if denom <= 0:
        return 1.0, 0.0, sig
    s = float(np.dot(d, raw) / denom)
    r2 = 1 - float(np.sum((raw - s * d) ** 2) / np.sum((raw - raw.mean()) ** 2))
    return 1 - r2, s, sig


def main():
    tt, uu, td, raw = load()
    # Compose kl/kd/kb directly so this can run before dark_affinity.json exists.
    kl = json.loads((OPTO / "0--LOVfast/fit_params.json").read_text())
    ki = json.loads((OPTO / "3--iLIDslow/fit_params.json").read_text())
    kk = {"kl1": kl["kl"], "kd1": kl["kd"], "kb1": kl["kb"], "kl2": ki["kl"], "kd2": ki["kd"], "kb2": ki["kb"]}

    res = minimize_scalar(lambda x: score(x, tt, uu, td, raw, kk)[0], bounds=(0.5, 8.0), method="bounded", options={"xatol": 1e-3})
    kd = float(res.x)
    loss, s, sig = score(kd, tt, uu, td, raw, kk)
    r2 = 1 - loss
    d, sig, ym = predict(kd, tt, uu, td, kk)
    rest = float(sig[0] / sig.max())
    floor = float(-s * sig[0])
    print(f"fitted Kd_dark / pool = {kd:.4f}")
    print(f"  scale s            = {s:.4f}")
    print(f"  R^2                = {r2:.4f}   (previous model, offset in y0: 0.9695)")
    print(f"  resting / peak     = {rest:.3f}   observed {raw[0] - raw.min():.4f}/" f"{raw.max() - raw.min():.4f} = {(raw[0] - raw.min()) / (raw.max() - raw.min()):.3f}")
    print(f"  dense floor        = {floor:+.4f}  observed {raw.min():+.4f}")
    print(f"  min state value    = {ym.min():+.3e}")
    for name, idx in (("A", (0, 1, 5, 6, 8, 9)), ("B", (2, 3, 5, 6, 7, 8, 9)), ("C", (4, 7, 8, 9))):
        tot = sum(ym[i] for i in idx)
        print(f"  {name} conserved to    {np.max(np.abs(tot - tot[0])):.2e}")
    if "--dry-run" in sys.argv:
        print("\ndry run -- nothing written")
        return 0
    OUT.write_text(
        json.dumps(
            {
                "kd_dark_over_pool": round(kd, 4),
                "kb2d_equals_kb2": True,
                "r2": round(r2, 4),
                "scale": round(s, 4),
                "resting_fraction_predicted": round(rest, 4),
                "fitted_on": str(RECORDING.relative_to(REPO)),
                "n_points": int(len(td)),
                "note": (
                    "dF/F0 normalisation removes a constant dark complex, so this cannot be " "fitted on the iLID biosensor traces; the decoder is the only recording " "that carries the information."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
