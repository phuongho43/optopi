import numpy as np
from scipy.integrate import ode

# Integrator settings. These are the ORIGINAL values, deliberately retained -- see below.
#
# Accuracy was audited in November 2026 against solve_ivp's LSODA and Radau at 1e-9.
# Findings, for anyone tempted to change these:
#
#   * These settings are accurate where it matters. On the FM-average sweep they agree
#     with the converged reference to 1.8e-3 (dense) and 8.5e-4 (sparse). On the
#     time-course traces the largest departure is ~40% of the output range, but it is
#     confined to 1-9 samples of ~1550, all at the instant light switches on, where the
#     curve is near-vertical. Rendered against the data the curves are indistinguishable.
#
#   * They leave states slightly negative (about -3.6e-5 across the reported sweeps), which is
#     the size of atol. This is integrator noise, not a modelling error: the structural
#     source of negative concentrations was the (AiBaC - AiBaC0) term, corrected below.
#
#   * Tightening is NOT a safe local edit. vode/bdf reports "Repeated error test
#     failures" and then silently returns a badly wrong solution rather than raising. At
#     rtol = 1e-10 with max_step = 0.1 it claimed the sparse FM peak sat at 0.20 Hz
#     instead of 0.042 Hz. rtol = 1e-7 / atol = 1e-9 / max_step = 0.02 was tested and
#     does hold every state above -4e-8 with 45x better trace accuracy, but it still
#     fails inside the swept-parameter regimes used by analysis/plot_optopi_levers.py,
#     where kd is multiplied up and the system gets stiffer.
#
# The right fix is a convergence check that detects the failure and retries, not new
# constants here. Until that exists, these values are the ones whose behaviour is known.
RTOL = 1e-5
ATOL = 1e-5
MAX_STEP = 0.1


def sim_lov(tt, y0, uu, kk):
    def model(t, y):
        [Ai, Aa, B, AB] = y
        kl = kk["kl"]
        kd = kk["kd"]
        kb = kk["kb"]
        v = 0 if u != 0 else 1
        dAi = -u * kl * Ai + v * kd * Aa - kb * Ai * B
        dAa = u * kl * Ai - v * kd * Aa + u * kl * AB
        dB = -kb * Ai * B + u * kl * AB
        dAB = kb * Ai * B - u * kl * AB
        return [dAi, dAa, dB, dAB]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=RTOL, atol=ATOL, max_step=MAX_STEP)
    solver.set_initial_value(y0)
    sol_t = [tt[0]]
    sol_y = [y0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_y.append(solver.y)
    return np.array(sol_t), np.array(sol_y).T


def sim_lid(tt, y0, uu, kk):
    def model(t, y):
        [Ai, Aa, B, AB] = y
        kl = kk["kl"]
        kd = kk["kd"]
        kb = kk["kb"]
        v = 0 if u != 0 else 1
        dAi = -u * kl * Ai + v * kd * Aa + v * kd * AB
        dAa = u * kl * Ai - v * kd * Aa - kb * Aa * B
        dB = -kb * Aa * B + v * kd * AB
        dAB = kb * Aa * B - v * kd * AB
        return [dAi, dAa, dB, dAB]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=RTOL, atol=ATOL, max_step=MAX_STEP)
    solver.set_initial_value(y0)
    sol_t = [tt[0]]
    sol_y = [y0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_y.append(solver.y)
    return np.array(sol_t), np.array(sol_y).T


def sparser_signal(ym):
    """The observed ddFP signal of the sparse decoder: AiBaC + AiBiC.

    [A] sits on LOV and [B] on sspB, so ANY complex holding the two together fluoresces --
    the lit ternary complex and the dark-state one alike. Read it through this function
    rather than as ``ym[-1]``: the sweeps in screen_3decoder.py and screen_crosstalk.py used
    that positional convention, and adding a species at the end of the state vector would
    have silently redirected them to the wrong state while still returning plausible curves.
    """
    return ym[8] + ym[9]


def sparser_dark_equilibrium(kk, total=1.0):
    """The resting state: LOV bound to Zdk, sspB partitioned by the dark affinity.

    Solved, not relaxed. In darkness Bi can never become Ba (that needs light) and AiBi can
    never come apart (that needs light too), so every A and B ends up as AiBi and the only
    equilibrium left is AiBi + C <-> AiBiC. With equimolar pools T and Kd = kd2d / kb2d:

        kb2d (T - x)^2 = kd2d x   =>   x^2 - (2T + Kd) x + T^2 = 0

    taking the root below T. Relaxing numerically instead looks equivalent and is not: the
    A + B association is second order with no dark reverse, so free Ai decays only as
    1/(kb1 T t) and still sits at ~1e-4 after 3000 s. tests/test_model_invariants.py caught
    that as a dark state drifting by 2e-5 -- small, but it would have put a slow creep under
    every sweep that starts from rest.

    Depends only on kb2d and kd2d, not on kd1 or kd2, which enter through terms that are
    zero in the dark -- so one call serves a whole reversion sweep.
    """
    kd = kk["kd2d"] / kk["kb2d"]
    b = 2.0 * total + kd
    x = (b - np.sqrt(b * b - 4.0 * total * total)) / 2.0
    free = total - x
    #        Ai   Aa   Bi   Ba   C     AiBi  AiBa BaC  AiBaC AiBiC
    return [0.0, 0.0, 0.0, 0.0, free, free, 0.0, 0.0, 0.0, float(x)]


def sim_sparser(tt, y0, uu, kk):
    """Sparse decoder: LOV(A) sequesters a Zdk-iLID fusion(B), which binds sspB(C) when lit.

    Ten states: Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC, AiBiC.

    Two terms were corrected in November 2026 after a reviewer ran this code, and a tenth
    species was added for the reason the second correction exposed. The measured effect of
    each is given below:

    1. Dissociation of AiBi under light released *active* iLID (``+ u * kl1 * AiBi`` in
       ``dBa``). Light breaking the LOV-Zdk bond says nothing about the iLID photostate,
       so the released species is inactive iLID and the term belongs in ``dBi``. As
       written, iLID could activate with its own photoconversion rate kl2 set to zero,
       and kl2 had almost no influence on the model at all (0.8% of the FM curve's
       range; 82% after this correction).

    2. Ternary-complex dissociation was ``v * kd2 * (AiBaC - AiBaC0)``, referenced to the
       complex's *initial* value. Below AiBaC0 the term changes sign and the reaction
       runs backwards, generating ternary complex while consuming C and AiBi; it drove
       four of the nine states negative under the fitted parameters. It is now
       ``v * kd2 * AiBaC``, which keeps every state non-negative.

    3. AiBiC, the ternary complex with iLID in its DARK state, is new. Correcting (2)
       showed what that term had been doing: the sparse-decoder recording rests 11% of its
       range ABOVE the level it reaches under dense input, so there is ternary complex
       present in the dark, and this model had no reaction that could make any -- only
       *active* iLID bound sspB. The resting complex had therefore been supplied as an
       initial condition, and ``- AiBaC0`` pinned it there so it could not decay. With the
       reference removed it decayed at kd2 (tau = 117 s), which is the baseline droop that
       appeared in Figure 2J. iLID/sspB has a well-documented dark-state affinity, so the
       honest fix is a reaction, not an initial condition:

           AiBi + C  <-- kd2d / kb2d -->  AiBiC      dark-state binding
           AiBiC  -- u * kl2 -->  AiBaC              photoconversion in place
           AiBaC  -- v * kd2 -->  AiBiC              iLID reverts, sspB stays bound
           AiBiC  -- u * kl1 -->  Aa + Bi + C        LOV released; the weak complex falls apart

       The resting level is now an equilibrium of the model instead of a number read off
       the data, and the sub-baseline excursion under dense input follows from the
       mechanism: continuous light keeps LOV off Zdk, which destroys the dark complex.
       Measured on the Figure 2J recording: R^2 0.970 -> 0.987, resting fraction 0.282
       predicted against 0.270 observed, and the dense floor -0.1118 against -0.1132. The
       parameter count is unchanged -- the previous version took the pool size and
       AiBaC(0) from the recording; this one takes the scale and kd2d/kb2d.

    Two approximations are retained deliberately rather than overlooked:

      * iLID bound to LOV cannot be photoconverted in place in the BINARY complex (there is
        no ``u * kl2 * AiBi -> AiBa`` term, nor its dark reverse). Adding that pair was
        tested and changes the FM-average curve by 0.01% of its range.
      * BaC reverts to Bi + C while AiBaC reverts to AiBiC -- the free lit complex falls
        apart, the LOV-bound one stays bound. The consistent eleven-state form, with BiC as
        well, was built and tested: R^2 0.985 against 0.987, a resting fraction of 0.322
        against 0.270, and states reaching -1.9e-5 against -1.5e-9. BaC is a transient,
        low-occupancy intermediate, so the simpler form is the one kept.

    Mass is conserved for all three species to ~3e-15, and tests/test_model_invariants.py
    checks that on every run -- a missing gain term in the eleven-state prototype leaked 15%
    of the B and C pools while still returning R^2 = 0.982, which no fit statistic would
    have shown.
    """

    def model(t, y):
        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC, AiBiC] = y
        kl1 = kk["kl1"]
        kd1 = kk["kd1"]
        kb1 = kk["kb1"]
        kl2 = kk["kl2"]
        kd2 = kk["kd2"]
        kb2 = kk["kb2"]
        kb2d = kk["kb2d"]
        kd2d = kk["kd2d"]
        v = 0 if u != 0 else 1
        dAi = -u * kl1 * Ai + v * kd1 * Aa - kb1 * Ai * Bi - kb1 * Ai * Ba - kb1 * Ai * BaC
        dAa = u * kl1 * Ai - v * kd1 * Aa + u * kl1 * (AiBi + AiBa + AiBaC + AiBiC)
        dBi = -u * kl2 * Bi + v * kd2 * Ba - kb1 * Ai * Bi + v * kd2 * BaC + u * kl1 * AiBi + u * kl1 * AiBiC
        dBa = u * kl2 * Bi - v * kd2 * Ba - kb1 * Ai * Ba - kb2 * Ba * C + u * kl1 * AiBa
        dC = -kb2 * Ba * C - kb2 * AiBa * C + v * kd2 * BaC - kb2d * AiBi * C + kd2d * AiBiC + u * kl1 * AiBiC
        dAiBi = kb1 * Ai * Bi - u * kl1 * AiBi - kb2d * AiBi * C + kd2d * AiBiC
        dAiBa = kb1 * Ai * Ba - kb2 * AiBa * C - u * kl1 * AiBa
        dBaC = kb2 * Ba * C - kb1 * Ai * BaC - v * kd2 * BaC + u * kl1 * AiBaC
        dAiBaC = kb2 * AiBa * C + kb1 * Ai * BaC - v * kd2 * AiBaC - u * kl1 * AiBaC + u * kl2 * AiBiC
        dAiBiC = v * kd2 * AiBaC + kb2d * AiBi * C - kd2d * AiBiC - u * kl2 * AiBiC - u * kl1 * AiBiC
        return [dAi, dAa, dBi, dBa, dC, dAiBi, dAiBa, dBaC, dAiBaC, dAiBiC]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=RTOL, atol=ATOL, max_step=MAX_STEP)
    solver.set_initial_value(y0)
    sol_t = [tt[0]]
    sol_y = [y0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_y.append(solver.y)
    return np.array(sol_t), np.array(sol_y).T
