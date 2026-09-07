"""
scripts / exp01_feasibility.py
================================
课题可行性验证实验：谱判据 是否 预测 经典可模拟性（H1）。

流程：
  1. 用多种特征映射 + 不同参数，生成大量样本 → 得到多个量子核
  2. 计算每个核的谱判据（effective_rank、spectral_entropy、concentration 等）
  3. 用 RFF / Nyström 测量"经典可模拟性"（逼近误差）
  4. 计算谱判据与可模拟性之间的 Spearman 相关，判断 H1 是否成立
  5. 散点图可视化

输出：
  - results/exp01_correlation.csv
  - figures/exp01_scatter.png
  - 控制台报告
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 设置中文字体 + 负号显示
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from feature_maps import encode_dataset, fidelity_kernel_from_embeddings, FEATURE_MAPS
from spectral_criteria import compute_all_criteria, effective_rank, spectral_entropy, concentration_index
from classical_baselines import RandomFourierFeatures, nystrom_approximation

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def make_dataset(n_samples, d, seed, mode="gauss"):
    """构造可控合成数据集。mode 控制数据结构复杂度。"""
    rng = np.random.default_rng(seed)
    if mode == "gauss":
        X = rng.normal(0, 0.5, size=(n_samples, d))
    elif mode == "cluster":
        # 聚类结构（含非线性）
        centers = rng.normal(0, 1.0, size=(8, d))
        labels = rng.integers(0, 8, size=n_samples)
        X = centers[labels] + rng.normal(0, 0.15, size=(n_samples, d))
    elif mode == "sin":
        t = rng.uniform(-np.pi, np.pi, size=n_samples)
        X = np.column_stack([np.sin(t * (i + 1)) + 0.1 * rng.normal(size=n_samples) for i in range(d)])
    else:
        raise ValueError(mode)
    return X


def run_single_config(fm_name, n_qubits, depth, X, seed):
    """对单个特征映射配置：编码 → 核 → 判据 + 可模拟性。"""
    params = {}
    if fm_name == "variational":
        params["entangle"] = 1.0
    elif fm_name == "high_dim":
        params["freq_scale"] = 1.5

    emb = encode_dataset(X, fm_name, n_qubits, depth=depth, **params)
    K = fidelity_kernel_from_embeddings(emb)

    # 谱判据
    crit = compute_all_criteria(K)

    # 经典可模拟性：RFF 逼近误差 + Nyström 误差
    d = X.shape[1]
    # RFF 训练（选接近量子核宽度的 gamma）
    rff_seed = seed
    rff_gamma = 0.5
    rff = RandomFourierFeatures(n_features=min(60, len(X)), gamma=rff_gamma, seed=rff_seed)
    rff.fit(X)
    rff_err = rff.approximate_target(K, X)

    _, nys_err = nystrom_approximation(K, n_landmark=min(12, len(X)), seed=seed)

    return {
        "fm": fm_name,
        "n_qubits": n_qubits,
        "depth": depth,
        "n_samples": len(X),
        "n_features": d,
        "seed": seed,
        "rff_error": rff_err,
        "nystrom_error": nys_err,
        **crit,
    }


def main():
    print("=" * 70)
    print("实验 01：谱判据 vs 经典可模拟性（H1 可行性验证）")
    print("=" * 70)

    # 实验矩阵：多种特征映射 × 多种深度 × 多种数据
    records = []
    n_samples = 40
    d = 4
    seeds = [0, 1, 2, 3]  # 每种配置用多个 seed 减少波动

    # 固定 n_qubits（因为 d=4，用 n_qubits=4 或 5）
    for fm_name in FEATURE_MAPS:
        for depth in [1, 2, 3]:
            for seed in seeds:
                X = make_dataset(n_samples, d, seed, mode="gauss")
                rec = run_single_config(fm_name, max(4, d), depth, X, seed)
                records.append(rec)

    # 额外：变分特征映射弱纠缠扫描
    for ent in [0.0, 0.3, 0.6, 1.0]:
        for seed in seeds:
            X = make_dataset(n_samples, d, seed, mode="gauss")
            emb = encode_dataset(X, "variational", max(4, d), depth=2, entangle=ent)
            K = fidelity_kernel_from_embeddings(emb)
            crit = compute_all_criteria(K)
            rff = RandomFourierFeatures(n_features=min(60, len(X)), gamma=0.5, seed=seed)
            rff.fit(X)
            rff_err = rff.approximate_target(K, X)
            _, nys_err = nystrom_approximation(K, n_landmark=min(12, len(X)), seed=seed)
            records.append({
                "fm": f"variational_ent{ent}",
                "n_qubits": max(4, d),
                "depth": 2,
                "n_samples": n_samples,
                "n_features": d,
                "seed": seed,
                "rff_error": rff_err,
                "nystrom_error": nys_err,
                **crit,
            })

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS, "exp01_correlation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[保存] {out_csv} （{len(df)} 条记录）")

    # === 相关性分析 H1 ===
    print("\n--- H1：谱判据 与 经典可模拟性（rff_error）的 Spearman 相关 ---")
    criteria_keys = [
        "effective_rank", "spectral_entropy", "concentration_index",
        "rank_ratio", "top5_energy", "gap_ratio", "participation_ratio",
    ]
    corr_rows = []
    for key in criteria_keys:
        mask = df[key].notna()
        if mask.sum() < 5:
            continue
        rho, p = spearmanr(df.loc[mask, key], df.loc[mask, "rff_error"])
        corr_rows.append({"criterion": key, "spearman_rho": rho, "p_value": p})
        print(f"  {key:22s}  rho={rho:+.3f}   p={p:.4g}")
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(RESULTS, "exp01_correlation_summary.csv"), index=False)

    # === H1 关键结论 ===
    print("\n--- 关键观察 ---")
    eff = df["effective_rank"]
    rff = df["rff_error"]
    print(f"  effective_rank 范围: [{eff.min():.3f}, {eff.max():.3f}]")
    print(f"  rff_error 范围:     [{rff.min():.3f}, {rff.max():.3f}]")

    # 分桶：有效秩高 vs 低的 RFF 误差
    low_mask = eff <= np.median(eff)
    high_mask = eff > np.median(eff)
    print(f"  有效秩低组(≤median) 平均 rff_error = {rff[low_mask].mean():.4f} (n={low_mask.sum()})")
    print(f"  有效秩高组(>median) 平均 rff_error = {rff[high_mask].mean():.4f} (n={high_mask.sum()})")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"angle": "tab:blue", "iqp": "tab:red", "variational": "tab:green", "high_dim": "tab:orange"}

    # 图1: eff_rank vs rff_error
    for fm, c in colors.items():
        sub = df[df["fm"].str.startswith(fm)]
        axes[0].scatter(sub["effective_rank"], sub["rff_error"], c=c, label=fm, alpha=0.7, s=40)
    axes[0].set_xlabel("Effective rank")
    axes[0].set_ylabel("RFF approximation error")
    axes[0].set_title("Spectral criterion vs classical simulability")
    axes[0].legend()

    # 图2: spectral_entropy vs rff_error
    for fm, c in colors.items():
        sub = df[df["fm"].str.startswith(fm)]
        axes[1].scatter(sub["spectral_entropy"], sub["rff_error"], c=c, label=None, alpha=0.7, s=40)
    axes[1].set_xlabel("Spectral entropy")
    axes[1].set_ylabel("RFF approximation error")
    axes[1].set_title("Spectral entropy vs simulability")

    # 图3: 各特征映射的平均谱（eff_rank 箱线/均值）
    fm_groups = df.groupby("fm")["effective_rank"].mean().sort_values()
    axes[2].bar(range(len(fm_groups)), fm_groups.values)
    axes[2].set_xticks(range(len(fm_groups)))
    axes[2].set_xticklabels(fm_groups.index, rotation=45, ha="right")
    axes[2].set_ylabel("Mean effective_rank")
    axes[2].set_title("Spectral structure by feature map")

    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp01_scatter.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    # === 结论输出 ===
    print("\n" + "=" * 70)
    print("结论（H1 验证）:")
    print("=" * 70)
    best_rho = corr_df.loc[corr_df["spearman_rho"].abs().idxmax()]
    print(f"  最强相关判据: {best_rho['criterion']} (rho={best_rho['spearman_rho']:+.3f}, p={best_rho['p_value']:.4g})")
    if best_rho["spearman_rho"] < 0 and best_rho["p_value"] < 0.05:
        print("  => 谱判据（有效秩）与经典可模拟性呈显著负相关：")
        print("     有效秩越低（谱越集中），RFF 越易逼近（越可经典模拟）→ H1 初步成立")
    else:
        print("  => 相关性未达显著，需进一步分析（可能需调整判据或扩大实验）")


if __name__ == "__main__":
    main()
