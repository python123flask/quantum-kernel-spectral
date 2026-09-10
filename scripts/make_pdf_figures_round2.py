"""
scripts / make_pdf_figures_round2.py
=====================================
第二轮图：exp06-09b 新结果的 PDF 矢量图。
  fig4_scaling.pdf    规模稳健性：rank~rff 随 qubit 数 + 真实数据优势 vs 经典难度
  fig5_alignment.pdf  核-目标对齐：全局显著 vs 控制难度后归零
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures", "pdf")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.7, "lines.linewidth": 1.1,
})


def fig4():
    df6 = pd.read_csv(os.path.join(RESULTS, "exp06_scaling.csv"))
    df10 = pd.read_csv(os.path.join(RESULTS, "exp10_scaling16.csv"))   # 16 qubit 扫描
    df7 = pd.read_csv(os.path.join(RESULTS, "exp07_real_data.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # 左：rank~rff 随规模（含 16 qubit，与正文完全对应）
    ns, rhos = [], []
    for n in [8, 12]:
        sub = df6[df6["n_qubits"] == n]
        r, _ = spearmanr(sub["effective_rank"], sub["rff_error"])
        ns.append(n); rhos.append(r)
    # 5 qubit 值来自 exp01（主实验）
    df1 = pd.read_csv(os.path.join(RESULTS, "exp01_correlation.csv"))
    r5, _ = spearmanr(df1["effective_rank"], df1["rff_error"])
    r16, _ = spearmanr(df10["effective_rank"], df10["rff_error"])
    ns = [5] + ns + [16]; rhos = [r5] + rhos + [r16]
    axes[0].plot(ns, rhos, "o-", color="#1f77b4", ms=5)
    for x, y in zip(ns, rhos):
        axes[0].annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)
    axes[0].set_xlabel("number of qubits $n$")
    axes[0].set_ylabel("$\\rho$(effective rank, RFF error)")
    axes[0].set_title("H1 strengthens with system size", fontsize=8.5)
    axes[0].set_ylim(0.5, 0.95)
    axes[0].set_xticks([5, 8, 12, 16])

    # 右：真实数据：经典基线 vs 量子优势
    for ds, c in zip(["iris", "wine", "breast_cancer", "digits"],
                     ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]):
        sub = df7[df7["dataset"] == ds]
        axes[1].scatter(sub["best_classical"], sub["quantum_advantage"],
                        s=18, alpha=0.8, color=c, label=ds)
    axes[1].axhline(0, color="0.4", ls="--", lw=0.8)
    axes[1].set_xlabel("best classical test accuracy")
    axes[1].set_ylabel("quantum advantage")
    axes[1].set_title("Real data: no advantage where classical saturates", fontsize=8.5)
    axes[1].set_xlim(0.92, 1.01)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig4_scaling.pdf"))
    plt.close(fig)
    print("[ok] fig4_scaling.pdf")


def fig5():
    df = pd.read_csv(os.path.join(RESULTS, "exp09b_alignment.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # 左：adv ~ TA（全局，按任务着色）
    colors = {"harmonic": "#ff7f0e", "periodic_coupled": "#1f77b4",
              "cross_high": "#d62728", "quantum_friendly": "#9467bd",
              "noisy_harmonic": "#2ca02c", "real_breast_cancer": "#8c564b"}
    for task, c in colors.items():
        sub = df[df["task"] == task]
        axes[0].scatter(sub["TA"], sub["adv"], s=16, alpha=0.8, color=c, edgecolors="none")
    axes[0].axhline(0, color="0.4", ls="--", lw=0.8)
    r, p = spearmanr(df["TA"], df["adv"])
    axes[0].set_xlabel("kernel-target alignment TA")
    axes[0].set_ylabel("quantum advantage")
    axes[0].set_title(f"TA predicts advantage: $\\rho$={r:+.2f} ($p$={p:.1e})", fontsize=8.5)

    # 右：控制难度后残差 ~ TA（归零）
    x_r = rankdata(df["best_classical"]); y_r = rankdata(df["adv"])
    lr = LinearRegression().fit(x_r.reshape(-1, 1), y_r)
    resid = y_r - lr.predict(x_r.reshape(-1, 1))
    for task, c in colors.items():
        sub = df[df["task"] == task].copy()
        sub["resid"] = resid[list(df[df["task"] == task].index)]
        axes[1].scatter(sub["TA"], sub["resid"], s=16, alpha=0.8, color=c, edgecolors="none")
    r2, p2 = spearmanr(df["TA"], resid)
    axes[1].set_xlabel("kernel-target alignment TA")
    axes[1].set_ylabel("advantage residual\n(rank-transformed)")
    axes[1].set_title(f"Residual vanishes: $\\rho$={r2:+.3f} ($p$={p2:.2f})", fontsize=8.5)
    axes[1].axhline(0, color="0.4", ls=":", lw=0.8)

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT, "fig5_alignment.pdf"))
    plt.close(fig)
    print("[ok] fig5_alignment.pdf")


if __name__ == "__main__":
    fig4()
    fig5()
    print("done ->", OUT)
