"""
scripts / exp05_data_dependence.py
==================================
补充实验：谱判据的数据依赖性（H4）。

研究问题：同一量子特征映射在不同数据分布上，其核谱结构（有效秩/谱熵）
变化有多大？"有效秩是否数据依赖"的答案是：__
若依赖性强，则任何"全局谱判据"都必须在数据上下文中使用 → 引出数据依赖判据。

方法：
  1. 固定特征映射（angle/iqp/variational/high_dim 各取 1-2 个代表性配置）
  2. 在不同数据分布上（高斯/聚类/周期/均匀/真实乳腺癌）
  3. 计算谱判据，报告跨数据分布的变异系数 CV 与分布
  4. 检验"数据内在维度"（PCA 有效维度）与谱判据的相关

输出：
  - results/exp05_data_dependence.csv
  - figures/exp05_data_dependence.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tasks import make_task
from feature_maps import encode_dataset, fidelity_kernel_from_embeddings
from spectral_criteria import compute_all_criteria, effective_rank, spectral_entropy

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def data_distributions(n, d, seed):
    """返回多种数据分布：(name, X)。"""
    rng = np.random.default_rng(seed)
    out = {}
    # 1. 高斯
    out["gaussian"] = rng.normal(0, 1.0, size=(n, d))
    # 2. 聚类（8 中心）
    centers = rng.normal(0, 1.5, size=(8, d))
    labels = rng.integers(0, 8, size=n)
    out["clusters"] = centers[labels] + rng.normal(0, 0.3, size=(n, d))
    # 3. 周期结构（均匀 + 正弦变换）
    t = rng.uniform(-np.pi, np.pi, size=n)
    out["periodic"] = np.column_stack([np.sin((i + 1) * t) + 0.1 * rng.normal(size=n) for i in range(d)])
    # 4. 均匀
    out["uniform"] = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, d))
    # 5. 重尾（Student-t 近似）
    out["heavy_tail"] = rng.standard_t(df=3, size=(n, d)) / np.sqrt(3)
    # 6. 真实数据（乳腺癌 PCA）
    try:
        Xb, _ = make_task("real_breast_cancer", n, d, seed)
        out["real_breast"] = Xb
    except Exception as e:
        print(f"  [warn] 真实数据不可用: {e}")
    return out


def pca_effective_dim(X, threshold=0.9):
    """PCA 累计方差达到 threshold 所需的主成分数（数据内在维度代理）。"""
    pca = PCA()
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cum, threshold) + 1)


def main():
    print("=" * 70)
    print("实验 05：谱判据的数据依赖性（H4）")
    print("=" * 70)

    n_samples = 100
    d = 5
    n_qubits = 5
    seeds = [0, 1, 2, 3]

    configs = [
        ("angle_d2", "angle", {"depth": 2}),
        ("iqp_d2", "iqp", {"depth": 2}),
        ("var_d2_e1", "variational", {"depth": 2, "entangle": 1.0}),
        ("highd_d2_f15", "high_dim", {"depth": 2, "freq_scale": 1.5}),
        ("highd_d2_f30", "high_dim", {"depth": 2, "freq_scale": 3.0}),
    ]

    records = []
    for name, fm, kwargs in configs:
        for seed in seeds:
            dists = data_distributions(n_samples, d, seed)
            for dname, X in dists.items():
                Xs = StandardScaler().fit_transform(X)
                emb = encode_dataset(Xs, fm, n_qubits, **kwargs)
                K = fidelity_kernel_from_embeddings(emb)
                crit = compute_all_criteria(K)
                records.append({
                    "config": name, "fm": fm, "data_dist": dname,
                    "seed": seed, "pca_eff_dim": pca_effective_dim(Xs),
                    **crit,
                })

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS, "exp05_data_dependence.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[保存] {out_csv} （{len(df)} 条记录）")

    # ============ 分析 ============
    print("\n--- 同一映射在不同数据分布上的 effective_rank（均值 ± 标准差） ---")
    pivot = df.pivot_table(index="data_dist", columns="config", values="effective_rank", aggfunc="mean").round(3)
    print(pivot)

    print("\n--- 变异系数 CV（跨数据分布） ---")
    cv_rows = []
    for name, fm, kwargs in configs:
        sub = df[df["config"] == name]["effective_rank"].values
        cv = np.std(sub) / (np.mean(sub) + 1e-9)
        cv_rows.append({"config": name, "mean_rank": np.mean(sub), "std_rank": np.std(sub), "cv": cv})
        print(f"  {name:14s} mean={np.mean(sub):.3f}  std={np.std(sub):.3f}  CV={cv:.3f}")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(RESULTS, "exp05_cv_summary.csv"), index=False)

    print("\n--- 数据内在维度（PCA eff dim）与 effective_rank 的相关 ---")
    rho, p = spearmanr(df["pca_eff_dim"], df["effective_rank"])
    print(f"  rho={rho:+.3f} p={p:.4g}")

    # ============ 可视化 ============
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 图1: 各映射×数据分布的 effective_rank 热图式条图
    order = ["gaussian", "clusters", "periodic", "uniform", "heavy_tail", "real_breast"]
    order = [d for d in order if d in df["data_dist"].unique()]
    width = 0.15
    x = np.arange(len(order))
    for i, name in enumerate([c[0] for c in configs]):
        vals = [df[(df["data_dist"] == d) & (df["config"] == name)]["effective_rank"].mean() for d in order]
        axes[0].bar(x + (i - len(configs) / 2 + 0.5) * width, vals, width, label=name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Effective rank")
    axes[0].set_title("Same feature map, different data distributions")
    axes[0].legend(fontsize=8)

    # 图2: 数据内在维度 vs effective_rank
    for name in [c[0] for c in configs]:
        sub = df[df["config"] == name]
        axes[1].scatter(sub["pca_eff_dim"], sub["effective_rank"], label=name, alpha=0.7, s=35)
    axes[1].set_xlabel("Data intrinsic dimension (PCA 90% var)")
    axes[1].set_ylabel("Effective rank")
    axes[1].set_title("Data intrinsic dimension vs spectral rank")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp05_data_dependence.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    # ============ 结论 ============
    print("\n" + "=" * 70)
    print("结论（exp05 摘要）:")
    print("=" * 70)
    print("  - CV 越大，谱判据越数据依赖 → 需要数据依赖判据。")
    print("  - 若 pca_eff_dim 与 effective_rank 正相关显著，则判据可部分由数据维度解释。")


if __name__ == "__main__":
    main()
