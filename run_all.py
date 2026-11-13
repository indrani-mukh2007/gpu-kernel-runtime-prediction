import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_generator import make_dataset
from src.models import eval_t4_models
from src.ablation import run_ablation
from src.transfer import eval_transfer
from src.scheduler import eval_scheduler


def save_figures(res_t4, abl_df, trans_df, sched_df):
    os.makedirs("plots", exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    models = res_t4["Model"].values
    x = np.arange(len(models))
    w = 0.25

    b1 = ax.bar(x - w, res_t4["Overall_MAPE_pct"], w, label="Overall", color="steelblue")
    b2 = ax.bar(x, res_t4["Compute_Bound_MAPE_pct"], w, label="Compute-Bound (sdpa)", color="seagreen")
    b3 = ax.bar(x + w, res_t4["Memory_Bound_MAPE_pct"], w, label="Memory-Bound (rmsnorm)", color="indianred")

    ax.set_ylabel("MAPE (%)")
    ax.set_title("Held-Out Operation Runtime Error (NVIDIA T4)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 38)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    for grp in (b1, b2, b3):
        for bar in grp:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig("plots/fig1_model_mape_comparison.png")
    plt.close(fig)

    sub = abl_df[abl_df["Ablated_Feature_Group"] != "None (Full Model)"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    y_pos = np.arange(len(sub))
    h_bar = 0.36

    ax.barh(y_pos - h_bar / 2, sub["Compute_Bound_MAPE_pct"], h_bar, label="Compute-Bound", color="seagreen")
    ax.barh(y_pos + h_bar / 2, sub["Memory_Bound_MAPE_pct"], h_bar, label="Memory-Bound", color="indianred")
    ax.axvline(
        res_t4.loc[0, "Overall_MAPE_pct"],
        color="steelblue",
        linestyle="--",
        linewidth=1.4,
        label=f"Full Model ({res_t4.loc[0, 'Overall_MAPE_pct']:.1f}%)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["Ablated_Feature_Group"])
    ax.invert_yaxis()
    ax.set_xlabel("Held-Out MAPE (%)")
    ax.set_title("Drop-Column Feature Ablation (NVIDIA T4)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig("plots/fig2_drop_column_ablation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fracs = trans_df["P100_Data_Fraction_pct"].values
    ax.plot(fracs, trans_df["Overall_MAPE_pct"], "o-", linewidth=2.0, color="steelblue", label="Overall P100 MAPE")
    ax.plot(fracs, trans_df["Compute_Bound_MAPE_pct"], "s--", linewidth=1.5, color="seagreen", label="Compute-Bound")
    ax.plot(fracs, trans_df["Memory_Bound_MAPE_pct"], "^--", linewidth=1.5, color="indianred", label="Memory-Bound")

    r10 = trans_df[trans_df["P100_Data_Fraction_pct"] == 10.0].iloc[0]
    ax.scatter([10.0], [r10["Overall_MAPE_pct"]], color="darkorange", s=80, zorder=5)
    ax.annotate(
        f"10% P100 Data\n({r10['Overall_MAPE_pct']:.1f}% MAPE)",
        xy=(10.0, r10["Overall_MAPE_pct"]),
        xytext=(18.0, r10["Overall_MAPE_pct"] + 4.5),
        arrowprops=dict(arrowstyle="->", color="0.2"),
    )

    ax.set_xlabel("P100 Calibration Data (%)")
    ax.set_ylabel("P100 Test MAPE (%)")
    ax.set_title("Cross-GPU Warm-Start Transfer (T4 -> P100)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig("plots/fig3_cross_gpu_transfer.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pols = sched_df["Policy"].values
    xp = np.arange(len(pols))
    cols = ["gray", "indianred", "peru", "steelblue", "seagreen"]

    bars = ax.bar(xp, sched_df["Mean_JCT_us"], width=0.52, color=cols, edgecolor="0.15", linewidth=0.7)
    ax.set_ylabel("Mean JCT (us)")
    ax.set_title("Shortest-Job-First (SJF) Queue Completion Time")
    ax.set_xticks(xp)
    ax.set_xticklabels(pols, rotation=15, ha="right")
    ax.set_ylim(0, max(sched_df["Mean_JCT_us"]) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, r_roof, c_orc in zip(
        bars,
        sched_df["JCT_Reduction_vs_Roofline_pct"],
        sched_df["Oracle_Gain_Captured_pct"],
    ):
        h = bar.get_height()
        ax.annotate(
            f"{h:.0f} us\n({-r_roof:+.1f}% vs Roof | {c_orc:.0f}% Orc)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    fig.tight_layout()
    fig.savefig("plots/fig4_sjf_scheduler_jct.png")
    plt.close(fig)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = make_dataset(n_total=2000, seed=42)
    df.to_csv("data/gpu_kernels_2000.csv", index=False)

    res_t4, _, _ = eval_t4_models(df)
    res_t4.to_csv("results/t4_model_comparison.csv", index=False)
    print("T4 Held-Out Comparison:")
    print(res_t4.to_string(index=False))

    abl_df = run_ablation(df, gpu="T4")
    abl_df.to_csv("results/ablation_results.csv", index=False)
    print("\nDrop-Column Ablation:")
    print(abl_df.to_string(index=False))

    trans_df = eval_transfer(df, seed=42)
    trans_df.to_csv("results/cross_gpu_transfer.csv", index=False)
    print("\nCross-GPU Transfer (T4 -> P100):")
    print(trans_df.to_string(index=False))

    sched_df = eval_scheduler(df, n_trials=300, q_size=40, seed=42)
    sched_df.to_csv("results/scheduler_jct_results.csv", index=False)
    print("\nSJF Scheduler Evaluation:")
    print(sched_df.to_string(index=False))

    save_figures(res_t4, abl_df, trans_df, sched_df)


if __name__ == "__main__":
    main()
