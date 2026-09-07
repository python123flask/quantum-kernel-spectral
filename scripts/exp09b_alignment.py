"""
scripts / exp09b_alignment.py
===============================
路径3补充：核-目标对齐（kernel-target alignment）作为任务-映射匹配量规。

Kübler et al. 2021 指出：量子核优势取决于目标函数是否落在 RKHS 中且难以经典表达。
核-目标对齐 TA(K, yy^T) 是标准任务-映射匹配度量。检验：
  1) TA 是否预测量子优势（全局）
  2) 控制 best_classical 后 TA 是否解释优势残差
  3) TA 是否解释任务内 rank→adv 的符号差异
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tasks import make_task, TASKS
from feature_maps import encode_dataset, fidelity_kernel_from_embeddings

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")

# 12 variant 参数（与 exp03 一致）
VARIANTS = {
    "angle_d1": ("angle", dict(depth=1)), "angle_d2": ("angle", dict(depth=2)),
    "angle_d3": ("angle", dict(depth=3)),
    "iqp_d1": ("iqp", dict(depth=1)), "iqp_d2": ("iqp", dict(depth=2)), "iqp_d3": ("iqp", dict(depth=3)),
    "var_d2_e03": ("variational", dict(depth=2, entangle=0.3)),
    "var_d2_e1": ("variational", dict(depth=2, entangle=1.0)),
    "var_d3_e1": ("variational", dict(depth=3, entangle=1.0)),
    "highd_d2_f15": ("high_dim", dict(depth=2, freq_scale=1.5)),
    "highd_d3_f15": ("high_dim", dict(depth=3, freq_scale=1.5)),
    "highd_d2_f30": ("high_dim", dict(depth=2, freq_scale=3.0)),
}


def target_alignment(K, y):
    """TA = <K, yy'>_F / (||K||_F ||yy'||_F)，中心化版（centered alignment 更常用）。"""
    n = len(y)
    Y = np.outer(y, y).astype(float)
    # centered kernel alignment（CCA 变体）
    Hc = np.eye(n) - np.ones((n, n)) / n
    Kc = Hc @ K @ Hc
    Yc = Hc @ Y @ Hc
    num = np.sum(Kc * Yc)
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float(num / (den + 1e-12))


def main():
    print("=" * 70)
    print("实验 09b：核-目标对齐作为任务-映射匹配量规")
    print("=" * 70)
    df = pd.read_csv(os.path.join(RESULTS, "exp03_threshold.csv"))
    n_samples, d, n_qubits, seeds = 120, 4, 5, [0, 1, 2]

    records = []
    for task in TASKS:
        for seed in seeds:
            X, y = make_task(task, n_samples, d, seed)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
            for vname, (fm, kw) in VARIANTS.items():
                emb = encode_dataset(X_tr, fm, n_qubits, **kw)
                K = fidelity_kernel_from_embeddings(emb)
                ta = target_alignment(K, y_tr)
                sub = df[(df["task"] == task) & (df["seed"] == seed) & (df["variant"] == vname)]
                if len(sub) == 0:
                    continue
                row = sub.iloc[0]
                records.append(dict(task=task, seed=seed, variant=vname, TA=ta,
                                    adv=row["quantum_advantage"],
                                    best_classical=row["best_classical"],
                                    rank_q=row["effective_rank"]))
    df2 = pd.DataFrame(records)
    print(f"重建 {len(df2)} 配置的 TA")

    print("\n--- 1) 全局相关性 ---")
    for x, label in [("TA", "TA"), ("best_classical", "best_classical"), ("rank_q", "rank_q")]:
        r, p = spearmanr(df2[x], df2["adv"])
        print(f"  adv ~ {label:16s} rho={r:+.3f} p={p:.3g}")

    print("\n--- 2) 控制 best_classical 后残差 ---")
    x_r = rankdata(df2["best_classical"]); y_r = rankdata(df2["adv"])
    lr = LinearRegression().fit(x_r.reshape(-1, 1), y_r)
    resid = y_r - lr.predict(x_r.reshape(-1, 1))
    df2["adv_resid"] = resid
    r, p = spearmanr(df2["TA"], df2["adv_resid"])
    print(f"  adv_resid ~ TA       rho={r:+.3f} p={p:.3g}")
    r, p = spearmanr(df2["rank_q"], df2["adv_resid"])
    print(f"  adv_resid ~ rank_q   rho={r:+.3f} p={p:.3g}")

    print("\n--- 3) 任务级 ---")
    rows = []
    for task in TASKS:
        sub = df2[df2["task"] == task]
        r_in, p_in = spearmanr(sub["rank_q"], sub["adv"])
        rows.append(dict(task=task, mean_TA=sub["TA"].mean(), mean_class=sub["best_classical"].mean(),
                         rho_within=r_in, p_within=p_in, mean_adv=sub["adv"].mean()))
    summ = pd.DataFrame(rows)
    print(summ.round(3).to_string(index=False))
    r, p = spearmanr(summ["mean_TA"], summ["rho_within"])
    print(f"\n  mean_TA ~ rho_within rho={r:+.3f} p={p:.3f}")

    out = os.path.join(RESULTS, "exp09b_alignment.csv")
    df2.to_csv(out, index=False)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
