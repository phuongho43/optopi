import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optopi.model import sim_lid, sim_sparser
from optopi.style import PALETTE, STYLE


def plot_model_fit(fig_fp, ty_model_df, ty_data_df, tu_df=None, figsize=(24, 16), leg_loc="best", palette=PALETTE, rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        class_labels = ["Data", "Model"]
        lw = rc_params["lines.linewidth"]
        if tu_df is not None:
            dt = tu_df["t"].diff().mean()
            for t in tu_df.loc[(tu_df["u"] > 0), "t"]:
                ax.axvspan(t, t + dt, color="#648FFF", alpha=0.8, lw=0)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, alpha=0.8, solid_capstyle="projecting"))
            class_labels.insert(0, "Input")
        sns.lineplot(ax=ax, data=ty_data_df, x="t", y="y", color=palette[0], linewidth=0, marker="o", markeredgewidth=0, zorder=2.3)
        sns.lineplot(ax=ax, data=ty_model_df, x="t", y="y", color=palette[1], zorder=2.2)
        line_handles = [
            mpl.lines.Line2D([], [], color=palette[0], marker="o", linewidth=0),
            mpl.lines.Line2D([], [], color=palette[1], linewidth=lw),
        ]
        handles.extend(line_handles)
        ax.legend(handles, class_labels, loc=leg_loc)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response (AU)")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    # ## Figure 3B, 3C, 3E, 3F ##
    # fig_fps = [
    #     "/home/phuong/data/phd-project/figures/fig_3b.png",
    #     "/home/phuong/data/phd-project/figures/fig_3c.png",
    #     "/home/phuong/data/phd-project/figures/fig_3e.png",
    #     "/home/phuong/data/phd-project/figures/fig_3f.png",
    # ]
    # class_dps = [
    #     "/home/phuong/data/phd-project/2--optopi/0--LOVfast",
    #     "/home/phuong/data/phd-project/2--optopi/1--LOVslow",
    #     "/home/phuong/data/phd-project/2--optopi/2--iLIDfast",
    #     "/home/phuong/data/phd-project/2--optopi/3--iLIDslow",
    # ]
    # if len(class_dps) != len(fig_fps):
    #     raise ValueError("len(class_dps)")
    # for c, class_dp in enumerate([Path(dp) for dp in class_dps]):
    #     fig_fp = fig_fps[c]
    #     k_json_fp = class_dp / "fit_params.json"
    #     with open(k_json_fp) as jf:
    #         kk = json.load(jf)
    #     y_csv_fp = class_dp / "y.csv"
    #     ty_data_df = pd.read_csv(y_csv_fp)
    #     yy = ty_data_df["y"].to_numpy()
    #     u_csv_fp = class_dp / "u.csv"
    #     tu_df = pd.read_csv(u_csv_fp)
    #     tu = tu_df["t"].to_numpy()
    #     uu = tu_df["u"].to_numpy()
    #     if "LOV" in str(class_dp):
    #         ode_model = sim_lov
    #         y0 = [0, 0, 0, -np.min(yy)]
    #         ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    #     elif "LID" in str(class_dp):
    #         ode_model = sim_lid
    #         y0 = [np.max(yy), 0, np.max(yy), 0]
    #     else:
    #         raise ValueError("ode_model")
    #     tm, ym = ode_model(tu, y0, uu, kk)
    #     ty_model_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    #     palette = ["#34495E", "#2ECC71"]
    #     plot_model_fit(fig_fp, ty_model_df, ty_data_df, tu_df=tu_df, palette=palette)

    # ## Fetch Sparse Decoder Data ##
    # class_dp = Path("/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/")
    # results_dp = Path("/home/phuong/data/phd-project/2--optopi/4--sparse-decoder/")
    # results_dp.mkdir(parents=True, exist_ok=True)
    # rty_df = process_rtz_data(class_dp, csv_fn="results/y.csv", calc_fnc=calc_dF_F0)
    # rty_df = rty_df.loc[(rty_df["t"] < 155)]
    # y_csv_fp = results_dp / "y.csv"
    # ty_mean = rty_df.groupby("t", as_index=False)["y"].mean().to_csv(y_csv_fp, index=False)
    # t0 = rty_df["t"].iloc[0]
    # tf = rty_df["t"].iloc[-1]
    # rtu_df = process_rtz_data(class_dp, csv_fn="u.csv", calc_fnc=convert_u_ta_tb, calc_fnc_kwargs={"t0": t0, "tf": tf})
    # rtu_df = rtu_df.loc[(rtu_df["t"] < 155)]
    # u_csv_fp = results_dp / "u.csv"
    # tu_mean = rtu_df.groupby("t", as_index=False)["u"].mean().to_csv(u_csv_fp, index=False)

    # ## Figure 3G ##
    # fig_fp = Path("/home/phuong/data/phd-project/figures/fig_3g.png")
    # class_dp = Path("/home/phuong/data/phd-project/2--optopi/4--sparse-decoder/")
    # lovfast_dp = Path("/home/phuong/data/phd-project/2--optopi/0--LOVfast")
    # ilidslow_dp = Path("/home/phuong/data/phd-project/2--optopi/3--iLIDslow/")
    # k_lovfast_json_fp = lovfast_dp / "fit_params.json"
    # with open(k_lovfast_json_fp) as jf:
    #     kk_lovfast = json.load(jf)
    # k_ilidslow_json_fp = ilidslow_dp / "fit_params.json"
    # with open(k_ilidslow_json_fp) as jf:
    #     kk_ilidslow = json.load(jf)
    # kk = {
    #     "kl1": kk_lovfast["kl"],
    #     "kd1": kk_lovfast["kd"],
    #     "kb1": kk_lovfast["kb"],
    #     "kl2": kk_ilidslow["kl"],
    #     "kd2": kk_ilidslow["kd"],
    #     "kb2": kk_ilidslow["kb"],
    # }
    # y_csv_fp = class_dp / "y.csv"
    # ty_data_df = pd.read_csv(y_csv_fp)
    # yy = ty_data_df["y"].to_numpy()
    # u_csv_fp = class_dp / "u.csv"
    # tu_df = pd.read_csv(u_csv_fp)
    # tu = tu_df["t"].to_numpy()
    # uu = tu_df["u"].to_numpy()
    # ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    # yy = yy - np.min(yy)
    # y0 = [0, 0, 0, 0, np.max(yy), np.max(yy), 0, 0, yy[0]]
    # tm, ym = sim_sparser(tu, y0, uu, kk)
    # ty_model_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    # palette = ["#34495E", "#EA822C"]
    # plot_model_fit(fig_fp, ty_model_df, ty_data_df, tu_df=tu_df, palette=palette)

    ## Sim FM Ave Response for Dense and Sparse Decoders ##
    # Prep Dense Decoder Data
    ilidfast_dp = Path("/home/phuong/data/phd-project/2--optopi/2--iLIDfast/")
    k_ilidfast_json_fp = ilidfast_dp / "fit_params.json"
    with open(k_ilidfast_json_fp) as jf:
        kk_denser = json.load(jf)
    y_csv_fp = ilidfast_dp / "y.csv"
    ty_data_df = pd.read_csv(y_csv_fp)
    yy = ty_data_df["y"].to_numpy()
    y0_denser = [np.max(yy), 0, np.max(yy), 0]
    # Prep Sparse Decoder Data
    sparser_dp = Path("/home/phuong/data/phd-project/2--optopi/4--sparse-decoder/")
    lovfast_dp = Path("/home/phuong/data/phd-project/2--optopi/0--LOVfast")
    ilidslow_dp = Path("/home/phuong/data/phd-project/2--optopi/3--iLIDslow/")
    k_lovfast_json_fp = lovfast_dp / "fit_params.json"
    with open(k_lovfast_json_fp) as jf:
        kk_lovfast = json.load(jf)
    k_ilidslow_json_fp = ilidslow_dp / "fit_params.json"
    with open(k_ilidslow_json_fp) as jf:
        kk_ilidslow = json.load(jf)
    kk_sparser = {
        "kl1": kk_lovfast["kl"],
        "kd1": kk_lovfast["kd"],
        "kb1": kk_lovfast["kb"],
        "kl2": kk_ilidslow["kl"],
        "kd2": kk_ilidslow["kd"],
        "kb2": kk_ilidslow["kb"],
    }
    y_csv_fp = sparser_dp / "y.csv"
    ty_data_df = pd.read_csv(y_csv_fp)
    yy = ty_data_df["y"].to_numpy()
    ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    yy = yy - np.min(yy)
    y0_sparser = [0, 0, 0, 0, np.max(yy), np.max(yy), 0, 0, yy[0]]
    # Calc FM Response
    fm_ave_df = []
    tu = np.arange(0, 121, 1)
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 121]
    for period in periods:
        freq = 1 / period
        uu = np.zeros_like(tu)
        uu[period:301:period] = 1
        tm_denser, ym_denser = sim_lid(tu, y0_denser, uu, kk_denser)
        y_ave_denser = np.mean(ym_denser[-1])
        fm_ave_df.append({"c": 0, "t": freq, "y": y_ave_denser})
        tm_sparser, ym_sparser = sim_sparser(tu, y0_sparser, uu, kk_sparser)
        y_ave_sparser = np.mean(ym_sparser[-1])
        fm_ave_df.append({"c": 1, "t": freq, "y": y_ave_sparser})
    fm_ave_dp = Path("/home/phuong/data/phd-project/2--optopi/5--fm-ave")
    fm_ave_dp.mkdir(parents=True, exist_ok=True)
    fm_ave_csv_fp = fm_ave_dp / "y.csv"
    fm_ave_df = pd.DataFrame(fm_ave_df).to_csv(fm_ave_csv_fp, index=False)


if __name__ == "__main__":
    main()
