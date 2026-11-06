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


def sim_sparser(tt, y0, uu, kk):
    """Sparse decoder: LOV(A) sequesters a Zdk-iLID fusion(B), which binds sspB(C) when lit.

    Two terms were corrected in November 2026 after a reviewer ran this code. The
    measured effect of each is given below:

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

    Note on a route this model does not contain: iLID bound to LOV cannot be
    photoconverted in place (there is no ``u * kl2 * AiBi -> AiBa`` term, nor its dark
    reverse). Adding that pair was tested and changes the FM-average curve by 0.01% of
    its range, so it is deliberately omitted rather than overlooked.

    Mass is conserved for all three species to ~3e-15 over a run, before and after.
    """

    def model(t, y):
        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC] = y
        kl1 = kk["kl1"]
        kd1 = kk["kd1"]
        kb1 = kk["kb1"]
        kl2 = kk["kl2"]
        kd2 = kk["kd2"]
        kb2 = kk["kb2"]
        v = 0 if u != 0 else 1
        dAi = -u * kl1 * Ai + v * kd1 * Aa - kb1 * Ai * Bi - kb1 * Ai * Ba - kb1 * Ai * BaC
        dAa = u * kl1 * Ai - v * kd1 * Aa + u * kl1 * AiBi + u * kl1 * AiBa + u * kl1 * AiBaC
        dBi = -u * kl2 * Bi + v * kd2 * Ba - kb1 * Ai * Bi + v * kd2 * BaC + u * kl1 * AiBi
        dBa = u * kl2 * Bi - v * kd2 * Ba - kb1 * Ai * Ba - kb2 * Ba * C + u * kl1 * AiBa
        dC = -kb2 * Ba * C - kb2 * AiBa * C + v * kd2 * BaC + v * kd2 * AiBaC
        dAiBi = kb1 * Ai * Bi - u * kl1 * AiBi + v * kd2 * AiBaC
        dAiBa = kb1 * Ai * Ba - kb2 * AiBa * C - u * kl1 * AiBa
        dBaC = kb2 * Ba * C - kb1 * Ai * BaC - v * kd2 * BaC + u * kl1 * AiBaC
        dAiBaC = kb2 * AiBa * C + kb1 * Ai * BaC - v * kd2 * AiBaC - u * kl1 * AiBaC
        return [dAi, dAa, dBi, dBa, dC, dAiBi, dAiBa, dBaC, dAiBaC]

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
