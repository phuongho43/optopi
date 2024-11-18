import json
import os

import numpy as np
import pandas as pd
from lmfit import Parameters, fit_report, minimize
from prettytable import PrettyTable

from optopi.model import sim_lid, sim_lov
from optopi.utils import makedirs

Y_CSV_FP = "/home/phuong/projects/optopi/example/data/LID/I427V/y.csv"
U_CSV_FP = "/home/phuong/projects/optopi/example/data/LID/I427V/u.csv"
SAVE_DP = "/home/phuong/projects/optopi/example/model/LID/I427V/"
ODE_MODEL = sim_lid  # 'sim_lov' or 'sim_lid'


def obj_func_opto_switch(kk, ty, yy, ye, y0, tu, uu):
    """Objective function for fitting the LOV or iLID model to light stimulation response data.

    Args:
        kk (1 x 3 array): [kl, kd, kb] rate parameters
        ty (1 x I array): response data timepoints
        yy (1 x I array): dimerized state [AB] response at each timepoint in ty
        ye (1 x I array): uncertainty for yy at each timepoint in ty (e.g. SEM)
        y0 (1 x 3 array): [Ai0, Aa0, B0, AB0] initial values
        tu (1 x J array): stimuli data timepoints
        uu (1 x J array): light stimulation intensity [u] at each timepoint in tu

    Returns:
        residual (1 x I array): error between data vs model, scaled by the uncertainty of yy
    """
    tm, ym = ODE_MODEL(tu, y0, uu, kk)
    ym = ym[-1]  # get model response for [AB]
    ym = ym[np.isin(tu, ty)]  # get only model response at same timepoints as the yy data
    residual = (yy - ym) / ye
    return residual


def iter_cb(params, itrn, resid, *args):
    sse = np.sum(resid**2)
    table = PrettyTable()
    table.field_names = ["Name", "Value"]
    table.add_row(["i", f"{itrn}".rjust(15)])
    table.add_row(["SSE", f"{sse:.5f}".rjust(15)])
    for name, param in params.items():
        table.add_row([name, f"{param.value:.5f}".rjust(15)])
    print(table)


def main():
    y_df = pd.read_csv(Y_CSV_FP)
    y_df = y_df.groupby("t", as_index=False)["y"].agg(["mean", "sem"])
    ty = y_df["t"].to_numpy()
    yy = y_df["mean"].to_numpy()
    ye = y_df["sem"].to_numpy()
    u_df = pd.read_csv(U_CSV_FP)
    u_df = u_df.groupby("t", as_index=False)["u"].agg(["mean"])
    tu = u_df["t"].to_numpy()
    uu = u_df["mean"].to_numpy()
    if sim_lov == ODE_MODEL:
        y0 = [0, 0, 0, -np.min(yy)]
        yy = yy - np.min(yy)
    elif sim_lid == ODE_MODEL:
        y0 = [np.max(yy), 0, np.max(yy), 0]
    else:
        raise ValueError("ODE_MODEL")
    kk = Parameters()
    kk.add("kl", value=0.0, min=0.0, max=100.0)
    kk.add("kd", value=0.0, min=0.0, max=1.0)
    kk.add("kb", value=0.0, min=0.0, max=100.0)
    opt = minimize(
        fcn=obj_func_opto_switch,
        params=kk,
        method="differential_evolution",
        args=(ty, yy, ye, y0, tu, uu),
        tol=1e-5,
        iter_cb=iter_cb,
        init="halton",
        strategy="best1bin",
        max_iter=10000,
        pop_size=1000,
        polish=True,
    )
    print(fit_report(opt))
    makedirs(SAVE_DP)
    report_fp = os.path.join(SAVE_DP, "fit_report.txt")
    with open(report_fp, "w", encoding="utf8") as text_file:
        print(fit_report(opt), file=text_file)
    params_fp = os.path.join(SAVE_DP, "fit_params.json")
    data = {name: param.value for name, param in opt.params.items()}
    with open(params_fp, "w", encoding="utf8") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
