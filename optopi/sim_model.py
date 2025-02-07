import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optopi.fit_model import prep_tu_tye_data
from optopi.model import sim_lid, sim_lov, sim_sparser
from optopi.style import PALETTE, STYLE


def plot_model_fit(fig_fp, ty_data_df, ty_model_df, tu_data_df=None, figsize=(24, 16), leg_loc="best", palette=PALETTE, rc_params=STYLE):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        class_labels = ["Data", "Model"]
        lw = rc_params["lines.linewidth"]
        if tu_data_df is not None:
            dt = tu_data_df["t"].diff().mean()
            for t in tu_data_df.loc[(tu_data_df["u"] > 0), "t"]:
                ax.axvspan(t, t + dt, color="#648FFF", lw=2)
            handles.append(mpl.lines.Line2D([], [], color="#648FFF", lw=lw, ls=(0, (1, 0))))
            class_labels.insert(0, "Input")
        sns.lineplot(ax=ax, data=ty_data_df, x="t", y="y", color=palette[0], linewidth=0, marker="o", markeredgewidth=0, zorder=2.3)
        sns.lineplot(ax=ax, data=ty_model_df, x="t", y="y", color=palette[1], lw=lw, zorder=2.2)
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


def prep_and_sim_lov(class_dp, k_json_fp):
    tu_data_df, ty_data_df = prep_tu_tye_data(class_dp)
    tu = tu_data_df["t"].to_numpy()
    uu = tu_data_df["u"].to_numpy()
    yy = ty_data_df["y"].to_numpy()
    ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    y0 = [0, 0, 0, -np.min(yy)]
    yy = yy - np.min(yy)
    with open(k_json_fp) as jf:
        kk = json.load(jf)
    tm, ym = sim_lov(tu, y0, uu, kk)
    ty_model_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    return tu_data_df, ty_data_df, ty_model_df


def prep_and_sim_lid(class_dp, k_json_fp):
    tu_data_df, ty_data_df = prep_tu_tye_data(class_dp)
    tu = tu_data_df["t"].to_numpy()
    uu = tu_data_df["u"].to_numpy()
    yy = ty_data_df["y"].to_numpy()
    y0 = [np.max(yy), 0, np.max(yy), 0]
    with open(k_json_fp) as jf:
        kk = json.load(jf)
    tm, ym = sim_lid(tu, y0, uu, kk)
    ty_model_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    return tu_data_df, ty_data_df, ty_model_df


def prep_sparser_params(k_lovfast_json_fp, k_ilidslow_json_fp):
    with open(k_lovfast_json_fp) as jf:
        kk_lovfast = json.load(jf)
    with open(k_ilidslow_json_fp) as jf:
        kk_ilidslow = json.load(jf)
    kk = {
        "kl1": kk_lovfast["kl"],
        "kd1": kk_lovfast["kd"],
        "kb1": kk_lovfast["kb"],
        "kl2": kk_ilidslow["kl"],
        "kd2": kk_ilidslow["kd"],
        "kb2": kk_ilidslow["kb"],
    }
    return kk


def prep_and_sim_sparser(class_dp, k_lovfast_json_fp, k_ilidslow_json_fp):
    tu_data_df, ty_data_df = prep_tu_tye_data(class_dp)
    tu_data_df = tu_data_df.loc[(tu_data_df["t"] < 155)]
    tu = tu_data_df["t"].to_numpy()
    uu = tu_data_df["u"].to_numpy()
    ty_data_df = ty_data_df.loc[(ty_data_df["t"] < 155)]
    yy = ty_data_df["y"].to_numpy()
    ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    yy = yy - np.min(yy)
    y0 = [0, 0, 0, 0, np.max(yy), np.max(yy), 0, 0, yy[0]]
    kk = prep_sparser_params(k_lovfast_json_fp, k_ilidslow_json_fp)
    tm, ym = sim_sparser(tu, y0, uu, kk)
    ty_model_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    return tu_data_df, ty_data_df, ty_model_df


def prep_and_sim_fm_ave(ilidfast_class_dp, sparser_class_dp, ilidfast_k_json_fp, lovfast_k_json_fp, ilidslow_k_json_fp):
    # Prep Dense Decoder Data
    _, ty_data_df = prep_tu_tye_data(ilidfast_class_dp)
    yy = ty_data_df["y"].to_numpy()
    y0_denser = [np.max(yy), 0, np.max(yy), 0]
    with open(ilidfast_k_json_fp) as jf:
        kk_denser = json.load(jf)
    # Prep Sparse Decoder Data
    _, ty_data_df = prep_tu_tye_data(sparser_class_dp)
    yy = ty_data_df["y"].to_numpy()
    ty_data_df["y"] = ty_data_df["y"] - np.min(yy)
    yy = yy - np.min(yy)
    y0_sparser = [0, 0, 0, 0, np.max(yy), np.max(yy), 0, 0, yy[0]]
    kk_sparser = prep_sparser_params(lovfast_k_json_fp, ilidslow_k_json_fp)
    # Calc FM Ave Response
    fm_ave_df = []
    tu = np.arange(0, 121, 1)
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 121]
    for period in periods:
        freq = 1 / period
        uu = np.zeros_like(tu)
        uu[period:121:period] = 1
        tm_denser, ym_denser = sim_lid(tu, y0_denser, uu, kk_denser)
        tm_sparser, ym_sparser = sim_sparser(tu, y0_sparser, uu, kk_sparser)
        y_ave_denser = np.mean(ym_denser[-1])
        y_ave_sparser = np.mean(ym_sparser[-1])
        fm_ave_df.append({"c": 0, "t": freq, "y": y_ave_denser})
        fm_ave_df.append({"c": 1, "t": freq, "y": y_ave_sparser})
    fm_ave_df = pd.DataFrame(fm_ave_df)
    return fm_ave_df


def main():
    ## Figure 3B, 3C, 3E, 3F ##
    fig_fps = [
        "/home/phuong/data/phd-project/figures/fig_3b.png",
        "/home/phuong/data/phd-project/figures/fig_3c.png",
        "/home/phuong/data/phd-project/figures/fig_3e.png",
        "/home/phuong/data/phd-project/figures/fig_3f.png",
    ]
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
    for fig_fp, results_dp, class_dp in zip(fig_fps, results_dps, class_dps):
        fig_fp = Path(fig_fp)
        results_dp = Path(results_dp)
        class_dp = Path(class_dp)
        k_json_fp = results_dp / "fit_params.json"
        if "lov" in str(class_dp).lower():
            tu_data_df, ty_data_df, ty_model_df = prep_and_sim_lov(class_dp, k_json_fp)
        elif "lid" in str(class_dp).lower():
            tu_data_df, ty_data_df, ty_model_df = prep_and_sim_lid(class_dp, k_json_fp)
        palette = ["#34495E", "#2ECC71"]
        plot_model_fit(fig_fp, ty_data_df, ty_model_df, tu_data_df=tu_data_df, palette=palette)

    ## Figure 3G ##
    fig_fp = Path("/home/phuong/data/phd-project/figures/fig_3g.png")
    class_dp = Path("/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/")
    k_lovfast_json_fp = Path("/home/phuong/data/phd-project/2--optopi/0--LOVfast/fit_params.json")
    k_ilidslow_json_fp = Path("/home/phuong/data/phd-project/2--optopi/3--iLIDslow/fit_params.json")
    tu_data_df, ty_data_df, ty_model_df = prep_and_sim_sparser(class_dp, k_lovfast_json_fp, k_ilidslow_json_fp)
    palette = ["#34495E", "#EA822C"]
    plot_model_fit(fig_fp, ty_data_df, ty_model_df, tu_data_df=tu_data_df, palette=palette)

    ## Sim FM Ave Response for Dense and Sparse Decoders ##
    ilidfast_class_dp = Path("/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/")
    sparser_class_dp = Path("/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/")
    ilidfast_k_json_fp = Path("/home/phuong/data/phd-project/2--optopi/2--iLIDfast/fit_params.json")
    lovfast_k_json_fp = Path("/home/phuong/data/phd-project/2--optopi/0--LOVfast/fit_params.json")
    ilidslow_k_json_fp = Path("/home/phuong/data/phd-project/2--optopi/3--iLIDslow/fit_params.json")
    fm_ave_df = prep_and_sim_fm_ave(ilidfast_class_dp, sparser_class_dp, ilidfast_k_json_fp, lovfast_k_json_fp, ilidslow_k_json_fp)
    fm_ave_dp = Path("/home/phuong/data/phd-project/2--optopi/4--fm-ave-response")
    fm_ave_dp.mkdir(parents=True, exist_ok=True)
    fm_ave_csv_fp = fm_ave_dp / "fm_ave_response.csv"
    fm_ave_df.to_csv(fm_ave_csv_fp, index=False)


if __name__ == "__main__":
    main()
