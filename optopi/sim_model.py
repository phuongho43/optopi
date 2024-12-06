import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optopi.model import sim_lid, sim_lov, sim_sparser

CUSTOM_PALETTE = ["#648FFF", "#2ECC71", "#8069EC", "#EA822C", "#D143A4", "#F1C40F", "#34495E"]

CUSTOM_STYLE = {
    "image.cmap": "turbo",
    "figure.figsize": (24, 16),
    "text.color": "#212121",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 12,
    "axes.labelcolor": "#212121",
    "axes.labelweight": 600,
    "axes.linewidth": 6,
    "axes.edgecolor": "#212121",
    "grid.linewidth": 1,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "lines.linewidth": 10,
    "axes.labelsize": 72,
    "xtick.labelsize": 56,
    "ytick.labelsize": 56,
    "legend.fontsize": 56,
}


def plot_model_fit(fig_fp, k_json_fp, y_csv_fp, u_csv_fp, ode_model):
    with open(k_json_fp) as jf:
        kk = json.load(jf)
    yd_df = pd.read_csv(y_csv_fp)
    y_df = yd_df.groupby("t", as_index=False)["y"].agg(["mean"])
    yy = y_df["mean"].to_numpy()
    ud_df = pd.read_csv(u_csv_fp)
    u_df = ud_df.groupby("t", as_index=False)["u"].agg(["mean"])
    tu = u_df["t"].to_numpy()
    uu = u_df["mean"].to_numpy()
    if ode_model == sim_lov:
        y0 = [0, 0, 0, -np.min(yy)]
        yd_df["y"] = yd_df["y"] - np.min(yy)
        yy = yy - np.min(yy)
    elif ode_model == sim_lid:
        y0 = [np.max(yy), 0, np.max(yy), 0]
    else:
        raise ValueError("ode_model")
    tm, ym = ode_model(tu, y0, uu, kk)
    ym_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.lineplot(data=ym_df, x="t", y="y", ax=ax, color="#2ECC71", zorder=2.2)
        sns.lineplot(
            data=yd_df,
            x="t",
            y="y",
            ax=ax,
            estimator="mean",
            errorbar=("se", 1.96),
            linewidth=0,
            color="#34495E",
            marker="o",
            markersize=12,
            markeredgewidth=0,
            zorder=2.3,
        )
        ymin, ymax = ax.get_ylim()
        plt.vlines(tu[uu > 0], ymin=ymin, ymax=ymax, colors="#648FFF", linewidth=1, alpha=0.5, zorder=2.1)
        ax.set_ylim(ymin, ymax)
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", linewidth=8, alpha=0.8),
            mpl.lines.Line2D([], [], color="#34495E", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#2ECC71", linewidth=16),
        ]
        group_labels = ["Input", "Data", "Model"]
        ax.legend(
            handles,
            group_labels,
            loc="best",
            markerscale=4,
            frameon=True,
            shadow=False,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AU")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=200, bbox_inches="tight", transparent=False)
    plt.close("all")


def plot_sparser_pred(fig_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_csv_fp, u_csv_fp):
    with open(k_lovfast_json_fp) as jf:
        kk_lovfast = json.load(jf)
    with open(k_lidslow_json_fp) as jf:
        kk_lidslow = json.load(jf)
    kk = {
        "kl1": kk_lovfast["kl"],
        "kd1": kk_lovfast["kd"],
        "kb1": kk_lovfast["kb"],
        "kl2": kk_lidslow["kl"],
        "kd2": kk_lidslow["kd"],
        "kb2": kk_lidslow["kb"],
    }
    yd_df = pd.read_csv(y_csv_fp)
    y_df = yd_df.groupby("t", as_index=False)["y"].agg(["mean"])
    yy = y_df["mean"].to_numpy()
    ud_df = pd.read_csv(u_csv_fp)
    u_df = ud_df.groupby("t", as_index=False)["u"].agg(["mean"])
    tu = u_df["t"].to_numpy()
    uu = u_df["mean"].to_numpy()
    yd_df["y"] = yd_df["y"] - np.min(yy)
    yy = yy - np.min(yy)
    y0 = [0, 0, 0, 0, np.max(yy), np.max(yy), 0, 0, yy[0]]
    tm, ym = sim_sparser(tu, y0, uu, kk)
    ym_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.lineplot(data=ym_df, x="t", y="y", ax=ax, color="#2ECC71", zorder=2.2)
        sns.lineplot(
            data=yd_df,
            x="t",
            y="y",
            ax=ax,
            estimator="mean",
            errorbar=("se", 1.96),
            linewidth=0,
            color="#34495E",
            marker="o",
            markersize=12,
            markeredgewidth=0,
            zorder=2.3,
        )
        ymin, ymax = ax.get_ylim()
        plt.vlines(tu[uu > 0], ymin=ymin, ymax=ymax, colors="#648FFF", linewidth=1, alpha=0.5, zorder=2.1)
        ax.set_ylim(ymin, ymax)
        handles = [
            mpl.lines.Line2D([], [], color="#648FFF", linewidth=8, alpha=0.8),
            mpl.lines.Line2D([], [], color="#34495E", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#2ECC71", linewidth=16),
        ]
        group_labels = ["Input", "Data", "Model"]
        ax.legend(
            handles,
            group_labels,
            loc="best",
            markerscale=4,
            frameon=True,
            shadow=False,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AU")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=200, bbox_inches="tight", transparent=False)
    plt.close("all")


def predict_fm_response(fig_fp, k_lidfast_json_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_denser_csv_fp, y_sparser_csv_fp):
    # Dense Decoder
    with open(k_lidfast_json_fp) as jf:
        kk_lidfast = json.load(jf)
    kk_denser = kk_lidfast
    yd_denser_df = pd.read_csv(y_denser_csv_fp)
    y_denser_df = yd_denser_df.groupby("t", as_index=False)["y"].agg(["mean"])
    yy_denser = y_denser_df["mean"].to_numpy()
    y0_denser = [np.max(yy_denser), 0, np.max(yy_denser), 0]
    ave_denser = []
    # Sparse Decoder
    with open(k_lovfast_json_fp) as jf:
        kk_lovfast = json.load(jf)
    with open(k_lidslow_json_fp) as jf:
        kk_lidslow = json.load(jf)
    kk_sparser = {
        "kl1": kk_lovfast["kl"],
        "kd1": kk_lovfast["kd"],
        "kb1": kk_lovfast["kb"],
        "kl2": kk_lidslow["kl"],
        "kd2": kk_lidslow["kd"],
        "kb2": kk_lidslow["kb"],
    }
    yd_sparser_df = pd.read_csv(y_sparser_csv_fp)
    y_sparser_df = yd_sparser_df.groupby("t", as_index=False)["y"].agg(["mean"])
    yy_sparser = y_sparser_df["mean"].to_numpy()
    yy_sparser = yy_sparser - np.min(yy_sparser)
    y0_sparser = [0, 0, 0, 0, np.max(yy_sparser), np.max(yy_sparser), 0, 0, yy_sparser[0]]
    ave_sparser = []
    # Calc FM Response
    tu = np.arange(0, 301, 1)
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 301]
    periods = np.array(periods)
    freqs = 1 / periods
    for period in periods:
        uu = np.zeros_like(tu)
        uu[period:301:period] = 1
        tm_denser, ym_denser = sim_lid(tu, y0_denser, uu, kk_denser)
        ave_denser.append(np.mean(ym_denser[-1]))
        tm_sparser, ym_sparser = sim_sparser(tu, y0_sparser, uu, kk_sparser)
        ave_sparser.append(np.mean(ym_sparser[-1]))
    ave_denser = np.array(ave_denser)
    ave_sparser = np.array(ave_sparser)
    print(f"Dense Decoder FM Peak: {freqs[np.argmax(ave_denser)]} Hz")
    print(f"Sparse Decoder FM Peak: {freqs[np.argmax(ave_sparser)]} Hz")
    ave_denser_df = pd.DataFrame({"t": freqs, "y": ave_denser, "h": np.ones_like(freqs) * 0})
    ave_sparser_df = pd.DataFrame({"t": freqs, "y": ave_sparser, "h": np.ones_like(freqs) * 1})
    ave_df = pd.concat([ave_denser_df, ave_sparser_df], ignore_index=True)
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
        palette = ["#8069EC", "#EA822C"]
        sns.lineplot(data=ave_df, x="t", y="y", hue="h", ax=ax, palette=palette)
        ax.set_xlabel("FM Input (Hz)")
        ax.set_ylabel("Mean Ouput (AU)")
        ax.set_xscale("log")
        group_labels = ["Dense Decoder", "Sparse Decoder"]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            group_labels,
            loc="best",
            markerscale=4,
            frameon=True,
            shadow=False,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1,
        )
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=200, bbox_inches="tight", transparent=False)
    plt.close("all")


def main():
    optopi_root_dp = Path(__file__).resolve().parent.parent
    example_dp = optopi_root_dp / "example"

    # Plot comparison of model fit and corresponding data
    for prot in ["LOV", "LID"]:
        for mut in ["I427V", "V416I"]:
            fig_fp = example_dp / "sim_model" / prot / mut / "model-fit.png"
            fig_fp.parent.mkdir(parents=True, exist_ok=True)
            k_json_fp = example_dp / "fit_model" / prot / mut / "fit_params.json"
            y_csv_fp = example_dp / "data" / prot / mut / "y.csv"
            u_csv_fp = example_dp / "data" / prot / mut / "u.csv"
            ode_model = sim_lov if prot == "LOV" else sim_lid
            plot_model_fit(fig_fp, k_json_fp, y_csv_fp, u_csv_fp, ode_model)

    # Plot Sparse Decoder model prediction compared to corresponding data
    fig_fp = example_dp / "sim_model" / "sparse_decoder" / "prediction.png"
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    k_lovfast_json_fp = example_dp / "fit_model" / "LOV" / "I427V" / "fit_params.json"
    k_lidslow_json_fp = example_dp / "fit_model" / "LID" / "V416I" / "fit_params.json"
    y_csv_fp = example_dp / "data" / "sparse_decoder" / "y.csv"
    u_csv_fp = example_dp / "data" / "sparse_decoder" / "u.csv"
    plot_sparser_pred(fig_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_csv_fp, u_csv_fp)

    # Calculate FM response for dense and sparse decoders
    fig_fp = example_dp / "sim_model" / "fm-response.png"
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    k_lidfast_json_fp = example_dp / "fit_model" / "LID" / "I427V" / "fit_params.json"
    k_lovfast_json_fp = example_dp / "fit_model" / "LOV" / "I427V" / "fit_params.json"
    k_lidslow_json_fp = example_dp / "fit_model" / "LID" / "V416I" / "fit_params.json"
    y_denser_csv_fp = example_dp / "data" / "LID" / "I427V" / "y.csv"
    y_sparser_csv_fp = example_dp / "data" / "sparse_decoder" / "y.csv"
    predict_fm_response(fig_fp, k_lidfast_json_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_denser_csv_fp, y_sparser_csv_fp)


if __name__ == "__main__":
    main()
