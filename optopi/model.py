import numpy as np
from scipy.integrate import ode


def sim_lov(tt, y0, uu, kk):
    def model(t, y):
        [Ai, Aa, B, AB] = y
        kl = kk["kl"].value
        kd = kk["kd"].value
        kb = kk["kb"].value
        v = 0 if u != 0 else 1
        dAi = -u * kl * Ai + v * kd * Aa - kb * Ai * B
        dAa = u * kl * Ai - v * kd * Aa + u * kl * AB
        dB = -kb * Ai * B + u * kl * AB
        dAB = kb * Ai * B - u * kl * AB
        return [dAi, dAa, dB, dAB]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=1e-6, atol=1e-6, max_step=0.1)
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
        kl = kk["kl"].value
        kd = kk["kd"].value
        kb = kk["kb"].value
        v = 0 if u != 0 else 1
        dAi = -u * kl * Ai + v * kd * Aa + v * kd * AB
        dAa = u * kl * Ai - v * kd * Aa - kb * Aa * B
        dB = -kb * Aa * B + v * kd * AB
        dAB = kb * Aa * B - v * kd * AB
        return [dAi, dAa, dB, dAB]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=1e-6, atol=1e-6, max_step=0.1)
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
    def model(t, y):
        [Ai0, Aa0, Bi0, Ba0, C0, AiBi0, AiBa0, BaC0, AiBaC0] = y0
        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC] = y
        kl1 = kk["kl1"].value
        kd1 = kk["kd1"].value
        kb1 = kk["kb1"].value
        kl2 = kk["kl2"].value
        kd2 = kk["kd2"].value
        kb2 = kk["kb2"].value
        [kl1, kd1, kb1, kl2, kd2, kb2] = kk
        v = 0 if u != 0 else 1
        dAi = -u * kl1 * Ai + v * kd1 * Aa - kb1 * Ai * Bi - kb1 * Ai * Ba - kb1 * Ai * BaC
        dAa = u * kl1 * Ai - v * kd1 * Aa + u * kl1 * AiBi + u * kl1 * AiBa + u * kl1 * AiBaC
        dBi = -u * kl2 * Bi + v * kd2 * Ba - kb1 * Ai * Bi + v * kd2 * BaC
        dBa = u * kl2 * Bi - v * kd2 * Ba - kb1 * Ai * Ba - kb2 * Ba * C + u * kl1 * AiBi + u * kl1 * AiBa
        dC = -kb2 * Ba * C - kb2 * AiBa * C + v * kd2 * BaC + v * kd2 * (AiBaC - AiBaC0)
        dAiBi = kb1 * Ai * Bi - u * kl1 * AiBi + v * kd2 * (AiBaC - AiBaC0)
        dAiBa = kb1 * Ai * Ba - kb2 * AiBa * C - u * kl1 * AiBa
        dBaC = kb2 * Ba * C - kb1 * Ai * BaC - v * kd2 * BaC + u * kl1 * AiBaC
        dAiBaC = kb2 * AiBa * C + kb1 * Ai * BaC - v * kd2 * (AiBaC - AiBaC0) - u * kl1 * AiBaC
        return [dAi, dAa, dBi, dBa, dC, dAiBi, dAiBa, dBaC, dAiBaC]

    solver = ode(model)
    solver.set_integrator("vode", method="bdf", rtol=1e-6, atol=1e-6, max_step=0.1)
    solver.set_initial_value(y0)
    sol_t = [tt[0]]
    sol_y = [y0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_y.append(solver.y)
    return np.array(sol_t), np.array(sol_y).T
