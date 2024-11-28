import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optopi.model import sim_lid, sim_lov, sim_sparser

CUSTOM_STYLE = {
    "image.cmap": "turbo",
    "figure.figsize": (24, 16),
    "text.color": "#212121",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 12,
    "axes.labelcolor": "#212121",
    "axes.labelweight": 600,
    "axes.linewidth": 4,
    "axes.edgecolor": "#212121",
    "grid.linewidth": 4,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "lines.linewidth": 10,
    "axes.labelsize": 64,
    "xtick.labelsize": 48,
    "ytick.labelsize": 48,
    "legend.fontsize": 48,
}


def plot_model_fit(fig_fp, k_json_fp, y_csv_fp, u_csv_fp, ode_model):
    fig_fp_pl = Path(fig_fp)
    fig_fp_pl.parent.mkdir(parents=True, exist_ok=True)
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
            zorder=5,
        )
        sns.lineplot(data=ym_df, x="t", y="y", ax=ax, color="#648FFF", zorder=4)
        handles = [
            mpl.lines.Line2D([], [], color="#34495E", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#648FFF", linewidth=16),
        ]
        group_labels = ["Data", "Model"]
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
        fig.savefig(fig_fp, pad_inches=0.3, dpi=100, bbox_inches="tight", transparent=False)
    plt.close("all")


def sim_sparser(fig_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_csv_fp, u_csv_fp):
    fig_fp_pl = Path(fig_fp)
    fig_fp_pl.parent.mkdir(parents=True, exist_ok=True)
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
    tm, ym = ode_model(tu, y0, uu, kk)
    ym_df = pd.DataFrame({"t": tm, "y": ym[-1]})
    with plt.style.context(("seaborn-v0_8-whitegrid", CUSTOM_STYLE)):
        fig, ax = plt.subplots(figsize=(24, 20))
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
            zorder=5,
        )
        sns.lineplot(data=ym_df, x="t", y="y", ax=ax, color="#648FFF", zorder=4)
        handles = [
            mpl.lines.Line2D([], [], color="#34495E", marker="o", markersize=8, linewidth=0),
            mpl.lines.Line2D([], [], color="#648FFF", linewidth=16),
        ]
        group_labels = ["Data", "Model"]
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
        fig.savefig(fig_fp, pad_inches=0.3, dpi=100, bbox_inches="tight", transparent=False)
    plt.close("all")


# def plot_figure_s10c():
#     # Model for dense channel
#     # iLIDfast parameters
#     kl = 48.277
#     kd = 0.213
#     kb = 89.984
#     kk_dense = [kl, kd, kb]
#     dense_dp = "/home/phuong/protosignet/1--biosensor/1--training/2--iLID/0--I427V/results/"
#     dense_csv_fp = os.path.join(dense_dp, "y.csv")
#     y_dense_df = pd.read_csv(dense_csv_fp)
#     y_dense_df = y_dense_df.groupby("t", as_index=False)["y"].mean()
#     ty_dense = y_dense_df["t"].values
#     yy_dense = y_dense_df["y"].values
#     y0_dense = [np.max(yy_dense), 0, np.max(yy_dense), 0]
#     ave_dense = []
#     # Model for sparse channel
#     # LOVfast parameters
#     kl1 = 51.982
#     kd1 = 0.369
#     kb1 = 3.407
#     # iLIDslow parameters
#     kl2 = 14.997
#     kd2 = 0.008
#     kb2 = 39.708
#     kk_sparse = [kl1, kd1, kb1, kl2, kd2, kb2]
#     sparse_dp = "/home/phuong/protosignet/1--biosensor/4--validation/0--sparse_ch/results/"
#     sparse_csv_fp = os.path.join(sparse_dp, "y.csv")
#     y_sparse_df = pd.read_csv(sparse_csv_fp)
#     y_sparse_df = y_sparse_df.groupby("t", as_index=False)["y"].mean()
#     ty_sparse = y_sparse_df["t"].values
#     yy_sparse = y_sparse_df["y"].values
#     yy_sparse = yy_sparse - np.min(yy_sparse)
#     y0_sparse = [0, 0, 0, 0, np.max(yy_sparse), np.max(yy_sparse), 0, 0, yy_sparse[0]]
#     ave_sparse = []
#     # freq scan -----------------
#     tu = np.arange(0, 301, 1)
#     periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
#     periods = np.array(periods)
#     freqs = 1 / periods
#     for period in periods:
#         print(period)
#         uu = np.zeros_like(tu)
#         uu[period:301:period] = 1
#         tm_dense, Xm_dense = sim_ilid(tu, y0_dense, uu, kk_dense)
#         ym_dense = Xm_dense[-1]
#         ave_dense.append(np.mean(ym_dense))
#         tm_sparse, Xm_sparse = sim_sparser(tu, y0_sparse, uu, kk_sparse)
#         ym_sparse = Xm_sparse[-1]
#         ave_sparse.append(np.mean(ym_sparse))
#     ave_dense = np.array(ave_dense)
#     ave_sparse = np.array(ave_sparse)
#     print("max freq dense ch:", freqs[np.argmax(ave_dense)])
#     print("max freq sparse ch:", freqs[np.argmax(ave_sparse)])
#     ave_dense_df = pd.DataFrame({"t": freqs, "y": ave_dense, "h": np.ones_like(freqs) * 0})
#     ave_sparse_df = pd.DataFrame({"t": freqs, "y": ave_sparse, "h": np.ones_like(freqs) * 1})
#     ave_df = pd.concat([ave_dense_df, ave_sparse_df], ignore_index=True)
#     fig_fp = os.path.join("/home/phuong/protosignet/3--figures/", "fig_s10c.png")
#     group_labels = ["Dense Ch.", "Sparse Ch."]
#     palette = ["#785EF0", "#FE6100"]
#     with plt.style.context(("seaborn-whitegrid", custom_styles)):
#         fig, ax = plt.subplots(figsize=(32, 16))
#         sns.lineplot(data=ave_df, x="t", y="y", hue="h", ax=ax, palette=palette, lw=10)
#         ax.set_xlabel("BL Pulsing Frequency (Hz)")
#         ax.set_ylabel("Mean Output (AU)")
#         ax.set_xscale("log")
#         # ax.xaxis.set_ticks([0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
#         # ax.get_xaxis().set_major_formatter(ScalarFormatter())
#         group_labels = ["Dense Ch.", "Sparse Ch."]
#         handles, _ = ax.get_legend_handles_labels()
#         ax.legend(
#             handles,
#             group_labels,
#             loc="best",
#             markerscale=4,
#             frameon=True,
#             shadow=False,
#             handletextpad=0.2,
#             borderpad=0.2,
#             labelspacing=0.2,
#             handlelength=1,
#         )
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig_fp = "/home/phuong/protosignet/3--figures/fig_s10c.png"
#         fig.savefig(fig_fp, pad_inches=0.3, dpi=100, bbox_inches="tight", transparent=False)


def main():
    optopi_root_dp = Path(__file__).resolve().parent
    example_dp = optopi_root_dp / "example"

    # Plot comparison of model fit and corresponding data
    for prot in ["LOV", "LID"]:
        for mut in ["I427V", "V416I"]:
            fig_fp = example_dp / "sim_model" / prot / mut / "model-fit.png"
            k_json_fp = example_dp / "fit_model" / prot / mut / "fit_params.json"
            y_csv_fp = example_dp / "data" / prot / mut / "y.csv"
            u_csv_fp = example_dp / "data" / prot / mut / "u.csv"
            ode_model = sim_lov if prot == "LOV" else sim_lid
            plot_model_fit(fig_fp, k_json_fp, y_csv_fp, u_csv_fp, ode_model)

    # Plot Sparse Decoder model prediction compared to corresponding data
    fig_fp = example_dp / "sim_model" / "sparse_decoder" / "prediction.png"
    k_lovfast_json_fp = example_dp / "fit_model" / "LOV" / "I427V" / "fit_params.json"
    k_lidslow_json_fp = example_dp / "fit_model" / "LID" / "V416I" / "fit_params.json"
    y_csv_fp = example_dp / "data" / "sparse_decoder" / "y.csv"
    u_csv_fp = example_dp / "data" / "sparse_decoder" / "u.csv"
    sim_sparser(fig_fp, k_lovfast_json_fp, k_lidslow_json_fp, y_csv_fp, u_csv_fp)


if __name__ == "__main__":
    main()
