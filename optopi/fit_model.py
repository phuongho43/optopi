import json
from pathlib import Path

import numpy as np
import pandas as pd
from lmfit import Parameters, fit_report, minimize
from natsort import natsorted
from prettytable import PrettyTable

from optopi.model import sim_lid, sim_lov


def convert_u_ta_tb(u_ta_tb_df, t0, tf):
    u_ta = np.round(u_ta_tb_df["ta"].values, 1)
    u_tb = np.round(u_ta_tb_df["tb"].values, 1)
    tt = np.round(np.arange(t0, tf + 0.1, 0.1), 1)
    uu = np.zeros_like(tt)
    for ta, tb in zip(u_ta, u_tb):
        if ta > tt[-1]:
            continue
        if tb > tt[-1]:
            tb = tt[-1]
        ia = np.where(tt == ta)[0][0]
        ib = np.where(tt == tb)[0][0]
        uu[ia:ib] = 1.0
    tu_df = pd.DataFrame({"t": tt, "u": uu})
    return tu_df


def calc_dF_F0(y_df):
    F0 = y_df["y"].iloc[:5].mean()
    dF = y_df["y"] - F0
    y_df["y"] = dF / F0
    return y_df


def prep_rtz_data(class_dp, csv_fn="y.csv", calc_fnc=None, calc_fnc_kwargs=None):
    rt_ave = []
    rtz_df = []
    for r, rep_dp in enumerate([rep_dp for rep_dp in natsorted(Path(class_dp).glob("*")) if rep_dp.is_dir()]):
        csv_fp = rep_dp / csv_fn
        tz_df = pd.read_csv(csv_fp)
        calc_fnc_kwargs = {} if calc_fnc_kwargs is None else calc_fnc_kwargs
        tz_df = calc_fnc(tz_df, **calc_fnc_kwargs)
        tz_df["r"] = np.ones(len(tz_df), dtype=int) * r
        rtz_df.append(tz_df)
        rt_ave.append(tz_df["t"].values)
    for tz_df in rtz_df:
        tz_df["t"] = np.array(rt_ave).mean(axis=0)
    rtz_df = pd.concat(rtz_df)
    return rtz_df


def prep_tu_tye_data(class_dp):
    class_dp = Path(class_dp)
    rty_df = prep_rtz_data(class_dp, csv_fn="results/y.csv", calc_fnc=calc_dF_F0)
    t0 = rty_df["t"].iloc[0]
    tf = rty_df["t"].iloc[-1]
    rtu_df = prep_rtz_data(class_dp, csv_fn="u.csv", calc_fnc=convert_u_ta_tb, calc_fnc_kwargs={"t0": t0, "tf": tf})
    tu_df = rtu_df.groupby("t", as_index=False)["u"].mean()
    tye_df = rty_df.groupby("t", as_index=False)["y"].mean()
    ty_sem_df = rty_df.groupby("t", as_index=False)["y"].sem()
    tye_df["e"] = ty_sem_df["y"]
    return tu_df, tye_df


def obj_func_opto_switch(kk, ty, yy, ye, y0, tu, uu, ode_model):
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
    kk = {name: param.value for name, param in kk.items()}
    tm, ym = ode_model(tu, y0, uu, kk)
    ym = ym[-1]  # get model response for [AB]
    idx = abs(ty[:, None] - tm[None, :]).argmin(axis=-1)  # indices of matching values between ty and tu
    ym = ym[idx]
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


def optimize_params(results_dp, tu_df, tye_df, ode_model):
    tu = tu_df["t"].to_numpy()
    uu = tu_df["u"].to_numpy()
    ty = tye_df["t"].to_numpy()
    yy = tye_df["y"].to_numpy()
    ye = tye_df["e"].to_numpy()
    if ode_model == sim_lov:
        y0 = [0, 0, 0, -np.min(yy)]
        yy = yy - np.min(yy)
    elif ode_model == sim_lid:
        y0 = [np.max(yy), 0, np.max(yy), 0]
    else:
        raise ValueError("ode_model")
    kk = Parameters()
    kk.add("kl", value=0.0, min=0.0, max=100.0)
    kk.add("kd", value=0.0, min=0.0, max=1.0)
    kk.add("kb", value=0.0, min=0.0, max=100.0)
    obj_func_opto_switch(kk, ty, yy, ye, y0, tu, uu, ode_model)
    opt = minimize(
        fcn=obj_func_opto_switch,
        args=(ty, yy, ye, y0, tu, uu, ode_model),
        params=kk,
        method="differential_evolution",
        tol=1e-6,
        init="halton",
        strategy="best1bin",
        max_iter=10000,
        pop_size=1000,
        polish=True,
        iter_cb=iter_cb,
    )
    print(fit_report(opt))
    results_dp = Path(results_dp)
    report_fp = results_dp / "fit_report.txt"
    with open(report_fp, "w", encoding="utf8") as text_file:
        print(fit_report(opt), file=text_file)
    params_fp = results_dp / "fit_params.json"
    data = {name: param.value for name, param in opt.params.items()}
    with open(params_fp, "w", encoding="utf8") as json_file:
        json.dump(data, json_file, indent=4)


def main():
    results_dps = [
        "/home/phuong/data/phd-project/2--optopi/0--LOVfast",
        "/home/phuong/data/phd-project/2--optopi/1--LOVslow",
        "/home/phuong/data/phd-project/2--optopi/2--iLIDfast",
        "/home/phuong/data/phd-project/2--optopi/3--iLIDslow",
    ]
    class_dps = [
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
    ]
    for results_dp, class_dp in zip(results_dps, class_dps):
        results_dp = Path(results_dp)
        results_dp.mkdir(parents=True, exist_ok=True)
        tu_df, tye_df = prep_tu_tye_data(class_dp)
        ode_model = sim_lov if "LOV" in str(class_dp) else sim_lid
        optimize_params(results_dp=results_dp, tu_df=tu_df, tye_df=tye_df, ode_model=ode_model)


if __name__ == "__main__":
    main()
