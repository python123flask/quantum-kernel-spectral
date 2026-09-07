"""
scripts / exp09_fit_criterion.py
==================================
路径3：任务-映射谱匹配量规——解释"优势残差"与判据-优势符号不稳定。

问题：为何同一谱判据在不同任务内与优势的相关方向不同（cross_high 负、
noisy 正）？假设：优势不仅由经典基线难度决定，还受"任务需要的谱形状"
与"映射提供的谱形状"匹配度影响。

量规：
  M1（谱展宽比）= effective_rank(quantum kernel) / effective_rank(RBF kernel)
  M2（谱形对齐）= cos 相似度(eigvec? 不用；用归一化特征值向量的余弦)
    cos( e_q, e_r ) = <λ_q, λ_r> / (||λ_q|| ||λ_r||)，λ 为归一化谱

分析：
  1) 全局：adv ~ M1、adv ~ best_classical 对比
  2) 控制 best_classical 后 adv 残差 ~ M1（偏相关）
  3) 任务级：每任务的 mean M1 与该任务 rank→adv 符号的关系
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

from tasks import make_task, TASKS
from classical_baselines import rbf_kernel
from spectral_criteria import effective_rank, spectral_entropy

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")


def rbf_spectral_features(X, gamma=0.5):
    K = rbf_kernel(X, gamma=gamma)
    return effective_rank(K), spectral_entropy(K)


def normalized_spectrum(K):
    w = np.linalg.eigvalsh(K)
    w = np.sort(np.maximum(w, 0))[::-1]
    s = w.sum()
    return w / s if s > 0 else w


def cos_similarity_spectra(Kq, Kr):
    """归一化特征值向量的余弦相似度（谱形状对齐）。"""
    a, b = normalized_spectrum(Kq), normalized_spectrum(Kr)
    n = min(len(a), len(b))
    return float(np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))


def main():
    print("=" * 70)
    print("实验 09：任务-映射谱匹配量规（优势残差解释）")
    print("=" * 70)
    df = pd.read_csv(os.path.join(RESULTS, "exp03_threshold.csv"))
    print(f"载入 exp03: {len(df)} 配置")

    n_samples, d, seeds = 120, 4, [0, 1, 2]
    records = []
    for task in TASKS:
        for seed in seeds:
            X, y = make_task(task, n_samples, d, seed)
            rank_rbf, ent_rbf = rbf_spectral_features(X)
            Kr = rbf_kernel(X, gamma=0.5)
            sub = df[(df["task"] == task) & (df["seed"] == seed)]
            for _, row in sub.iterrows():
                # 重建该配置的量子核（同参数）以求谱形对齐——需要 variant 参数
                # 简化：用 exp03 已算出的 effective_rank；谱形对齐仅对可重建变体做
                records.append(dict(
                    task=task, seed=seed, variant=row["variant"],
                    adv=row["quantum_advantage"], best_classical=row["best_classical"],
                    rank_q=row["effective_rank"], rank_rbf=rank_rbf, ent_rbf=ent_rbf,
                    M1=row["effective_rank"] / max(rank_rbf, 1e-9),
                ))
    df2 = pd.DataFrame(records)
    print(f"构建匹配表: {len(df2)} 行")

    print("\n--- 1) 全局相关性 ---")
    for x, label in [("M1", "M1=rank_q/rank_rbf"), ("best_classical", "best_classical"),
                     ("rank_q", "rank_q")]:
        r, p = spearmanr(df2[x], df2["adv"])
        print(f"  adv ~ {label:24s} rho={r:+.3f} p={p:.3g}")

    print("\n--- 2) 控制 best_classical 后的偏相关（残差法）---")
    x_r = rankdata(df2["best_classical"]); y_r = rankdata(df2["adv"])
    lr = LinearRegression().fit(x_r.reshape(-1, 1), y_r)
    resid = y_r - lr.predict(x_r.reshape(-1, 1))
    df2["adv_resid"] = resid
    for x, label in [("M1", "M1"), ("rank_q", "rank_q"), ("rank_rbf", "rank_rbf")]:
        r, p = spearmanr(df2[x], df2["adv_resid"])
        print(f"  adv_resid ~ {label:12s} rho={r:+.3f} p={p:.3g}")

    print("\n--- 3) 任务级：M1、best_classical 与任务内 rank→adv 符号 ---")
    rows = []
    for task in TASKS:
        sub = df2[df2["task"] == task]
        rho_in, p_in = spearmanr(sub["rank_q"], sub["adv"])
        rows.append(dict(task=task, n=len(sub), mean_M1=sub["M1"].mean(),
                         mean_class=sub["best_classical"].mean(),
                         rho_within=rho_in, p_within=p_in))
    summ = pd.DataFrame(rows)
    print(summ.round(3).to_string(index=False))
    # M1 与 rho_within 的关系
    r, p = spearmanr(summ["mean_M1"], summ["rho_within"])
    print(f"\n  mean_M1 ~ rho_within rho={r:+.3f} p={p:.3f}  (n={len(summ)} 任务)")

    out = os.path.join(RESULTS, "exp09_fit_criterion.csv")
    df2.to_csv(out, index=False)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
