"""
scripts / exp03_supp_difficulty.py
===================================
exp03 补充分析：量子优势 与 经典基线难度 的关系。

核心问题：periodic_coupled 上经典基线 ~0.52（经典也很难）时量子核能打平，
而 cross_high 上经典基线 ~0.86 时量子全面落后。
检验：quantum_advantage 是否与 best_classical（经典难度代理）负相关。

输出：
  - results/exp03_difficulty_analysis.csv
  - figures/exp03_advantage_vs_difficulty.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")


def main():
    df = pd.read_csv(os.path.join(RESULTS, "exp03_threshold.csv"))

    # 1) 全数据：advantage vs best_classical
    rho_all, p_all = spearmanr(df["best_classical"], df["quantum_advantage"])
    print(f"全数据: advantage vs best_classical  rho={rho_all:+.3f}  p={p_all:.4g}")

    # 2) 按任务分层
    print("\n按任务分层:")
    rows = []
    for t in df["task"].unique():
        sub = df[df["task"] == t]
        rho, p = spearmanr(sub["best_classical"], sub["quantum_advantage"])
        rows.append({"task": t, "n": len(sub), "rho": rho, "p": p,
                     "mean_best_classical": sub["best_classical"].mean(),
                     "pos_frac": (sub["quantum_advantage"] > 0).mean()})
        print(f"  {t:22s} n={len(sub):3d} rho={rho:+.3f} p={p:.4g} mean_classical={sub['best_classical'].mean():.3f} pos_frac={(sub['quantum_advantage']>0).mean():.2f}")

    # 3) 任务级汇总：每个任务 (mean classic difficulty, mean advantage, pos_frac)
    task_summary = df.groupby("task").agg(
        mean_advantage=("quantum_advantage", "mean"),
        mean_classical=("best_classical", "mean"),
        pos_frac=("quantum_advantage", lambda s: (s > 0).mean()),
        mean_rank=("effective_rank", "mean"),
    ).round(4)
    print("\n任务级汇总:")
    print(task_summary)

    task_summary.to_csv(os.path.join(RESULTS, "exp03_task_summary.csv"))

    # 4) 可视化：advantage vs best_classical（按任务）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    tasks = df["task"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for i, t in enumerate(tasks):
        sub = df[df["task"] == t]
        axes[0].scatter(sub["best_classical"], sub["quantum_advantage"],
                        color=colors[i], alpha=0.75, s=45, label=t)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Best classical accuracy (task difficulty proxy)")
    axes[0].set_ylabel("Quantum advantage")
    axes[0].set_title("Advantage vs classical difficulty (per config)")
    axes[0].legend(fontsize=8)

    # 任务级：classical difficulty vs pos_frac / mean advantage
    ax = axes[1]
    ax.scatter(task_summary["mean_classical"], task_summary["mean_advantage"],
               s=120, c="tab:blue", label="mean advantage")
    ax2 = ax.twinx()
    ax2.scatter(task_summary["mean_classical"], task_summary["pos_frac"],
                s=120, c="tab:orange", marker="^", label="pos fraction")
    ax.set_xlabel("Mean best-classical accuracy (task difficulty)")
    ax.set_ylabel("Mean quantum advantage", color="tab:blue")
    ax2.set_ylabel("Fraction of positive-advantage configs", color="tab:orange")
    ax.set_title("Task-level: difficulty vs quantum opportunity")
    for i, t in enumerate(tasks):
        ax.annotate(t, (task_summary["mean_classical"][t], task_summary["mean_advantage"][t]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp03_advantage_vs_difficulty.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "exp03_difficulty_analysis.csv"), index=False)

    # 5) 结论
    print("\n" + "=" * 70)
    print("结论（exp03 补充）:")
    print("=" * 70)
    print(f"  全数据 advantage~classical_difficulty rho={rho_all:+.3f} p={p_all:.4g}")
    print("  - 若显著负相关：经典越难的任务，量子核越有机会 → '任务-映射匹配'假说。")


if __name__ == "__main__":
    main()
