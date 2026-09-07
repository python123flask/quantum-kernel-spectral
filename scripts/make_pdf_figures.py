"""
scripts / make_pdf_figures.py
==============================
从 results CSV 重绘论文核心图为期刊级 PDF 矢量图（Quantum 风格）。

输出（figures/pdf/）：
  fig1_difficulty.pdf   优势 vs 经典基线难度（左：全局散点；右：任务级单调）
  fig2_simulability.pdf 谱判据 vs RFF 逼近误差（左：有效秩；右：谱熵）
  fig3_freqscale.pdf    频率尺度旋钮对谱结构与可模拟性的完美单调效应

所有图均为矢量 PDF，直接用于投递稿件。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures", "pdf")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

TASK_COLORS = {
    "periodic_coupled": "#1f77b4",
    "noisy_harmonic": "#2ca02c",
    "harmonic": "#ff7f0e",
    "quantum_friendly": "#9467bd",
    "cross_high": "#d62728",
    "real_breast_cancer": "#8c564b",
}
TASK_LABELS = {
    "periodic_coupled": "periodic coupled",
    "noisy_harmonic": "noisy harmonic",
    "harmonic": "harmonic",
    "quantum_friendly": "quantum friendly",
    "cross_high": "cross-term",
    "real_breast_cancer": "breast cancer",
}


def fig1():
    df = pd.read_csv(os.path.join(RESULTS, "exp03_threshold.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for task, c in TASK_COLORS.items():
        sub = df[df["task"] == task]
        axes[0].scatter(sub["best_classical"], sub["quantum_advantage"],
                        s=14, alpha=0.75, color=c, edgecolors="none",
                        label=TASK_LABELS[task])
    axes[0].axhline(0, color="0.4", ls="--", lw=0.8)
    axes[0].set_xlabel("best classical test accuracy")
    axes[0].set_ylabel("quantum advantage")
    axes[0].set_xlim(0.3, 1.0)
    axes[0].legend(frameon=False, handletextpad=0.2, labelspacing=0.3,
                   loc="lower left", fontsize=8)

    # 右：任务级 pos_frac vs mean best classical
    agg = df.groupby("task").agg(
        mean_class=("best_classical", "mean"),
        pos_frac=("quantum_advantage", lambda s: (s > 0).mean()),
        mean_adv=("quantum_advantage", "mean"),
    ).reset_index().sort_values("mean_class")
    axes[1].plot(agg["mean_class"], agg["pos_frac"], "o-",
                 color="0.25", ms=5, lw=1.2)
    for _, r in agg.iterrows():
        axes[1].annotate(TASK_LABELS[r["task"]],
                         (r["mean_class"], r["pos_frac"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("task mean best classical accuracy")
    axes[1].set_ylabel("fraction of configs with\npositive advantage")
    axes[1].set_ylim(-0.03, 0.5)
    axes[1].set_xlim(0.48, 1.0)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig1_difficulty.pdf"))
    plt.close(fig)
    print("[ok] fig1_difficulty.pdf")


def fig2():
    df = pd.read_csv(os.path.join(RESULTS, "exp01_correlation.csv"))
    fm_colors = {"angle": "#1f77b4", "iqp": "#d62728",
                 "variational": "#2ca02c", "high_dim": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for fm, c in fm_colors.items():
        sub = df[df["fm"].str.startswith(fm)]
        axes[0].scatter(sub["effective_rank"], sub["rff_error"],
                        s=20, alpha=0.8, color=c, edgecolors="none", label=fm)
        axes[1].scatter(sub["spectral_entropy"], sub["rff_error"],
                        s=20, alpha=0.8, color=c, edgecolors="none")
    for ax in axes:
        ax.set_ylabel("RFF approximation error")
        ax.axhline(0.5, color="0.7", ls=":", lw=0.7)
    axes[0].set_xlabel("effective rank")
    axes[0].set_title("concentrated spectra are easier to simulate", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("spectral entropy (bits)")
    axes[1].set_title("spectral entropy vs simulability", fontsize=8.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig2_simulability.pdf"))
    plt.close(fig)
    print("[ok] fig2_simulability.pdf")


def fig3():
    df = pd.read_csv(os.path.join(RESULTS, "exp04_knob_sweep.csv"))
    sub = df[(df["knob"] == "freq_scale") & (df["depth"] == 2)].copy()
    sub = sub.groupby("value")[["effective_rank", "rff_error"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    axes[0].plot(sub["value"], sub["effective_rank"], "o-", color="#1f77b4", ms=4.5)
    axes[0].set_xlabel("frequency scale $s$")
    axes[0].set_ylabel("mean effective rank")
    axes[0].set_title("Fourier spread controls the spectrum", fontsize=8.5)
    axes[1].plot(sub["value"], sub["rff_error"], "s-", color="#d62728", ms=4.5)
    axes[1].set_xlabel("frequency scale $s$")
    axes[1].set_ylabel("mean RFF approximation error")
    axes[1].set_title("...and classical approximability", fontsize=8.5)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig3_freqscale.pdf"))
    plt.close(fig)
    print("[ok] fig3_freqscale.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("done ->", OUT)
