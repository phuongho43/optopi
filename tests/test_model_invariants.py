"""Invariants every photoswitch model must hold, checked on every CI run.

WRITTEN 2026-11-06. optopi/model.py asserts in prose that "mass is conserved for all three
species to ~3e-15" and that the corrected dissociation term "keeps every state
non-negative". Nothing tested either claim, and both had already been violated in this
file's history:

  * ``v * kd2 * (AiBaC - AiBaC0)`` changed sign below the complex's initial value and ran
    the reaction backwards, driving four of the nine states negative under the fitted
    parameters. It survived a reviewer, a release and a submission.
  * while prototyping a dark-binding extension on 2026-11-06, a single missing gain term
    (``BiC --u*kl2--> BaC`` was removed from one equation and never added to the other)
    leaked 15% of the B and C pools. The fit looked plausible -- R^2 = 0.982 -- and only
    the conservation sum showed it.

A fit statistic cannot see either defect: the system still integrates and still resembles
the data. A conservation sum sees both immediately, costs a second to compute, and does not
depend on any data file. So it belongs here rather than in a notebook.
"""

import numpy as np
import pytest

from optopi.model import (
    ATOL,
    sim_lid,
    sim_lov,
    sim_sparser,
    sparser_dark_equilibrium,
    sparser_signal,
)

# Conserved totals, as index groups into each model's state vector.
#   sim_lov / sim_lid:  [Ai, Aa, B, AB]
#   sim_sparser:        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC, AiBiC]
POOLS = {
    "sim_lov": {"A": (0, 1, 3), "B": (2, 3)},
    "sim_lid": {"A": (0, 1, 3), "B": (2, 3)},
    "sim_sparser": {"A": (0, 1, 5, 6, 8, 9), "B": (2, 3, 5, 6, 7, 8, 9), "C": (4, 7, 8, 9)},
}

KK_LOV = {"kl": 44.91, "kd": 0.3750, "kb": 2.660}  # LOVfast, as fitted
KK_LID = {"kl": 13.20, "kd": 0.008537, "kb": 33.16}  # iLIDslow, as fitted
KK_SPARSER = {"kl1": 44.91, "kd1": 0.3750, "kb1": 2.660, "kl2": 13.20, "kd2": 0.008537, "kb2": 33.16, "kb2d": 33.16, "kd2d": 2.5 * 33.16}  # dark affinity, Kd = 2.5 pools


def pulses(period, width=1.0, tmax=260.0, dt=0.1, dense_from=None):
    """A sparse pulse train, optionally followed by continuous light."""
    tt = np.round(np.arange(0.0, tmax + dt, dt), 1)
    uu = np.zeros_like(tt)
    for start in np.arange(60.0, dense_from or tmax, period):
        uu[(tt >= start) & (tt < start + width)] = 1.0
    if dense_from is not None:
        uu[tt >= dense_from] = 1.0
    return tt, uu


CASES = [
    # name,          sim,          y0,                              kk
    ("lov_dark_start", sim_lov, [1.0, 0.0, 1.0, 0.0], KK_LOV),
    ("lov_bound_start", sim_lov, [0.0, 0.0, 0.0, 1.0], KK_LOV),
    ("lid_dark_start", sim_lid, [1.0, 0.0, 1.0, 0.0], KK_LID),
    ("sparser_unbound_start", sim_sparser, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], KK_SPARSER),
    ("sparser_bound_start", sim_sparser, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], KK_SPARSER),
    # The resting ternary complex supplied by hand, which is what the shipped Figure 2J
    # initial condition did and what the (AiBaC - AiBaC0) term used to pin in place.
    ("sparser_resting_complex", sim_sparser, [0.0, 0.0, 0.0, 0.0, 0.44, 0.44, 0.0, 0.0, 0.0, 0.12], KK_SPARSER),
]
INPUTS = [("sparse_10s", {"period": 10.0}), ("sparse_then_dense", {"period": 10.0, "dense_from": 180.0}), ("dense_only", {"period": 0.1, "width": 0.1, "dense_from": 60.0})]


@pytest.mark.parametrize("name,sim,y0,kk", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("input_name,kwargs", INPUTS, ids=[i[0] for i in INPUTS])
def test_mass_is_conserved(name, sim, y0, kk, input_name, kwargs):
    tt, uu = pulses(**kwargs)
    _, ym = sim(tt, y0, uu, kk)
    for species, idx in POOLS[sim.__name__].items():
        total = sum(ym[i] for i in idx)
        drift = float(np.max(np.abs(total - total[0])))
        assert drift < 1e-9, f"{sim.__name__} does not conserve {species}: it drifts by {drift:.3e} " f"from {total[0]:.6f}. A term was added to one equation and not subtracted " f"from another."


@pytest.mark.parametrize("name,sim,y0,kk", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("input_name,kwargs", INPUTS, ids=[i[0] for i in INPUTS])
def test_states_stay_non_negative(name, sim, y0, kk, input_name, kwargs):
    tt, uu = pulses(**kwargs)
    _, ym = sim(tt, y0, uu, kk)
    worst = float(ym.min())
    # The tolerance is the integrator's atol: anything larger is structural, which is what
    # the (AiBaC - AiBaC0) term produced.
    assert worst > -1e-4, f"{sim.__name__} drives a state to {worst:.3e}, well past integrator noise. " f"A concentration cannot be negative; check for a term that changes sign."


def test_dark_equilibrium_is_stationary():
    """The resting state must be a steady state of the equations, not a number we chose.

    This is the check the model could not have passed before AiBiC existed. The decoder
    rests with ternary complex present, but the nine-state model had no reaction that could
    make any, so the resting complex had to be supplied as an initial condition -- and left
    to itself in the dark it decayed at kd2. Now it is an equilibrium: relax into it, leave
    the light off, and nothing moves.
    """
    y0 = sparser_dark_equilibrium(KK_SPARSER)
    tt = np.round(np.arange(0.0, 600.1, 0.1), 1)
    _, ym = sim_sparser(tt, y0, np.zeros_like(tt), KK_SPARSER)
    drift = float(np.max(np.abs(ym - np.array(y0)[:, None])))
    # The threshold is the integrator's absolute tolerance, not a round number: the analytic
    # point is an exact fixed point (max |dy/dt| = 3.6e-15) and LSODA at 1e-11 holds it to
    # 3.4e-13, so anything at the 1e-5 scale is vode's noise and anything above it is not.
    assert drift < 2 * ATOL, f"the dark equilibrium is not stationary: it moves by {drift:.3e}, more than the " f"integrator's tolerance of {ATOL:.0e} explains"
    assert sparser_signal(ym)[0] > 0.01, (
        "the dark equilibrium holds no ternary complex, so the model still cannot produce " "a resting signal and the sub-baseline dense response has nothing to come from"
    )


def test_resting_complex_is_destroyed_by_continuous_light():
    """Dense input must take the signal BELOW its dark level -- the whole reason for AiBiC.

    Continuous light keeps LOV off Zdk, so neither ternary complex can hold together and the
    observable falls to roughly zero, i.e. 100% of the resting level below baseline. That is
    the excursion the recording shows and the nine-state model could only reproduce by having
    the offset handed to it.
    """
    y0 = sparser_dark_equilibrium(KK_SPARSER)
    tt = np.round(np.arange(0.0, 300.1, 0.1), 1)
    uu = np.where(tt >= 60.0, 1.0, 0.0)
    _, ym = sim_sparser(tt, y0, uu, KK_SPARSER)
    sig = sparser_signal(ym)
    assert sig[-1] < 0.1 * sig[0], f"continuous light leaves {sig[-1] / sig[0]:.1%} of the resting complex; it should " f"destroy nearly all of it"
