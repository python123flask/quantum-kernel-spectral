"""
scripts / exp03_spectral_threshold.py
======================================
课题核心实验：量子泛化优势的"谱阈值窗口"扫描（H3 深化）。

研究问题：是否存在一个谱判据阈值 τ，使得
  - effective_rank(或谱熵) < τ 时，量子核几乎总是被经典核超越（可模拟/无优势）
  - effective_rank > τ 时，量子核可能展示真正的优势（优势窗口）

相比 exp02 的改进：
  1. 更丰富的任务集（谐波/耦合周期/高阶交叉/量子友好/噪声/真实数据）
  2. 经典基线更强更公平（RBF gamma 用验证集选参、RFF 大特征数、多项式核）
  3. 阈值检测（候选阈值扫描 + Mann-Whitney U + Cliff's delta + BH-FDR 校正）
  4. 记录任务经典难度（线性基线），分析优势是否与难度交互

输出：
  - results/exp03_threshold.csv
  - results/exp03_threshold_analysis.csv
  - figures/exp03_threshold_window.png
  - figures/exp03_advantage_vs_rank_by_task.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from tasks import make_task, TASKS
from feature_maps import encode_dataset, fidelity_kernel_from_embeddings
from spectral_criteria import compute_all_criteria
from classical_baselines import (
    rbf_kernel, linear_kernel, polynomial_kernel, RandomFourierFeatures
)

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

SMOKE = "--smoke" in sys.argv


# ---------------------------------------------------------------------------
# 特征映射变体（含"设计旋钮"）
# 每个变体是 (name_in_csv, fm, kwargs)
# ---------------------------------------------------------------------------
def variant_list():
    return [
        ("angle_d1", "angle", {"depth": 1}),
        ("angle_d2", "angle", {"depth": 2}),
        ("angle_d3", "angle", {"depth": 3}),
        ("iqp_d1", "iqp", {"depth": 1}),
        ("iqp_d2", "iqp", {"depth": 2}),
        ("iqp_d3", "iqp", {"depth": 3}),
        ("var_d2_e1", "variational", {"depth": 2, "entangle": 1.0}),
        ("var_d2_e03", "variational", {"depth": 2, "entangle": 0.3}),
        ("var_d3_e1", "variational", {"depth": 3, "entangle": 1.0}),
        ("highd_d2_f15", "high_dim", {"depth": 2, "freq_scale": 1.5}),
        ("highd_d2_f30", "high_dim", {"depth": 2, "freq_scale": 3.0}),
        ("highd_d3_f15", "high_dim", {"depth": 3, "freq_scale": 1.5}),
    ]


# ---------------------------------------------------------------------------
# 强经典基线（验证集选参）
# ---------------------------------------------------------------------------
def train_classical_baselines(X_tr, y_tr, X_te, y_te, seed):
    """RBF(gamma 验证集选参) + RFF + 线性 + 多项式，返回最佳经典精度和各基线精度。"""
    # 内部划分 train2 / val
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=seed)

    # 1) RBF：gamma 网格选参
    gammas = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    best_gamma, best_acc = None, -1
    for g in gammas:
        K_tr2 = rbf_kernel(X_tr2, gamma=g)
        K_v = rbf_kernel(X_val, X_tr2, gamma=g)
        m = SVC(kernel="precomputed")
        m.fit(K_tr2, y_tr2)
        acc = accuracy_score(y_val, m.predict(K_v))
        if acc > best_acc:
            best_acc, best_gamma = acc, g
    K_tr = rbf_kernel(X_tr, gamma=best_gamma)
    K_te = rbf_kernel(X_te, X_tr, gamma=best_gamma)
    m = SVC(kernel="precomputed")
    m.fit(K_tr, y_tr)
    acc_rbf = accuracy_score(y_te, m.predict(K_te))

    # 2) RFF：同 gamma，特征数较多
    n_feat = min(150, len(X_tr))
    rff = RandomFourierFeatures(n_features=n_feat, gamma=best_gamma, seed=seed)
    rff.fit(X_tr)
    K_tr_rff = rff.kernel(X_tr)
    K_te_rff = rff.kernel(X_te, X_tr)
    m = SVC(kernel="precomputed")
    m.fit(K_tr_rff, y_tr)
    acc_rff = accuracy_score(y_te, m.predict(K_te_rff))

    # 3) 线性
    K_tr_l = linear_kernel(X_tr)
    K_te_l = linear_kernel(X_te, X_tr)
    m = SVC(kernel="precomputed")
    m.fit(K_tr_l, y_tr)
    acc_linear = accuracy_score(y_te, m.predict(K_te_l))

    # 4) 多项式 (3 阶)
    K_tr_p = polynomial_kernel(X_tr, degree=3)
    K_te_p = polynomial_kernel(X_te, X_tr, degree=3)
    m = SVC(kernel="precomputed")
    m.fit(K_tr_p, y_tr)
    acc_poly = accuracy_score(y_te, m.predict(K_te_p))

    best_classical = max(acc_rbf, acc_rff, acc_linear, acc_poly)
    return {
        "acc_rbf": acc_rbf, "acc_rff": acc_rff,
        "acc_linear": acc_linear, "acc_poly": acc_poly,
        "best_classical": best_classical, "best_gamma": best_gamma,
    }


# ---------------------------------------------------------------------------
# 单配置运行
# ---------------------------------------------------------------------------
def run_config(task_name, variant_name, fm, kwargs, X, y, n_qubits, seed):
    # 划分 train/test（70/30）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if len(np.unique(y)) == 2 else None)

    # 经典基线
    base = train_classical_baselines(X_tr, y_tr, X_te, y_te, seed)

    # 量子核
    emb_tr = encode_dataset(X_tr, fm, n_qubits, **kwargs)
    emb_te = encode_dataset(X_te, fm, n_qubits, **kwargs)
    K_q_tr = fidelity_kernel_from_embeddings(emb_tr)
    G_te = emb_te @ emb_tr.conj().T
    K_q_te = np.abs(G_te) ** 2

    m = SVC(kernel="precomputed")
    m.fit(K_q_tr, y_tr)
    acc_q = accuracy_score(y_te, m.predict(K_q_te))

    crit = compute_all_criteria(K_q_tr)

    return {
        "task": task_name,
        "variant": variant_name,
        "feature_map": fm,
        "n_qubits": n_qubits,
        "seed": seed,
        "acc_quantum": acc_q,
        "quantum_advantage": acc_q - base["best_classical"],
        **base,
        **crit,
    }


# ---------------------------------------------------------------------------
# 阈值窗口分析
# ---------------------------------------------------------------------------
def threshold_analysis(df, criterion="effective_rank", n_bins=12):
    """
    对谱判据做分箱，检验"窗口"：
      1) 分箱均值曲线
      2) 候选阈值 τ（各分位点）扫描：高/低两组 Mann-Whitney U + Cliff's delta
      3) BH-FDR 校正
    返回：
      - bins_df: 每个箱的 (中位谱值, 优势均值, n, 正优势比例)
      - tau_df: 每个候选阈值的 (τ, u_p, cliff_delta, n_low, n_high, 高组优势均值)
    """
    vals = df[criterion].values
    adv = df["quantum_advantage"].values

    # 分箱
    bins = pd.qcut(df[criterion], q=min(n_bins, df[criterion].nunique()), duplicates="drop")
    bins_df = df.groupby(bins, observed=True).agg(
        n=("quantum_advantage", "size"),
        adv_mean=("quantum_advantage", "mean"),
        adv_std=("quantum_advantage", "std"),
        pos_frac=("quantum_advantage", lambda s: (s > 0).mean()),
    ).reset_index()
    bins_df.columns = ["bin", "n", "adv_mean", "adv_std", "pos_frac"]
    bins_df["criterion_med"] = [b.mid for b in bins_df["bin"]]

    # 候选阈值扫描
    quantiles = np.linspace(0.15, 0.85, 15)
    taus = np.quantile(vals, quantiles)
    tau_rows = []
    for tau in taus:
        low = adv[vals <= tau]
        high = adv[vals > tau]
        if len(low) < 5 or len(high) < 5:
            continue
        stat, p = mannwhitneyu(high, low, alternative="greater")
        # Cliff's delta
        r = rankdata(adv)
        # 直接计算 Cliff's delta: P(X>Y) - P(X<Y)
        n_hi, n_lo = len(high), len(low)
        cnt_greater = sum(1 for h in high for l in low if h > l)
        cnt_less = sum(1 for h in high for l in low if h < l)
        cliff = (cnt_greater - cnt_less) / (n_hi * n_lo)
        tau_rows.append({
            "tau": tau,
            "mannwhitney_p": p,
            "cliff_delta": cliff,
            "n_low": n_lo, "n_high": n_hi,
            "adv_mean_low": low.mean(), "adv_mean_high": high.mean(),
        })
    tau_df = pd.DataFrame(tau_rows)

    # BH-FDR 校正
    if len(tau_df) > 0:
        pvals = tau_df["mannwhitney_p"].values
        p_sorted_idx = np.argsort(pvals)
        m = len(pvals)
        adjusted = np.empty(m)
        for rank_i, idx in enumerate(p_sorted_idx):
            adjusted[idx] = min(1.0, pvals[idx] * m / (rank_i + 1))
        # 单调化
        for i in range(m - 2, -1, -1):
            j = p_sorted_idx[i]
            k = p_sorted_idx[i + 1]
            adjusted[j] = min(adjusted[j], adjusted[k])
        tau_df["fdr_p"] = adjusted

    return bins_df, tau_df


def main():
    print("=" * 70)
    print("实验 03：量子优势的谱阈值窗口扫描（H3 深化）")
    print("=" * 70)

    if SMOKE:
        print("[SMOKE 模式：只有少量配置]")

    n_samples = 120
    d = 4
    n_qubits = 5
    tasks = ["harmonic", "periodic_coupled", "cross_high", "quantum_friendly", "noisy_harmonic", "real_breast_cancer"]
    seeds = [0, 1, 2] if not SMOKE else [0]

    records = []
    total = len(tasks) * len(variant_list()) * len(seeds)
    n_done = 0
    for task_name in tasks:
        for seed in seeds:
            X, y = make_task(task_name, n_samples, d, seed)
            # 标准化（真实与合成统一）
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            for vname, fm, kwargs in variant_list():
                rec = run_config(task_name, vname, fm, kwargs, X, y, n_qubits, seed)
                records.append(rec)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  ... {n_done}/{total} 配置完成")

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS, "exp03_threshold.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[保存] {out_csv} （{len(df)} 条记录）")

    # ============ 汇总统计 ============
    print("\n--- 各任务上 quantum_advantage 统计 ---")
    g = df.groupby("task")["quantum_advantage"].agg(["mean", "std", "max", lambda s: (s > 0).mean()])
    g.columns = ["mean", "std", "max", "pos_frac"]
    print(g.round(4))

    print("\n--- 各变体（特征映射）平均 effective_rank / 优势 ---")
    v = df.groupby("variant")[["effective_rank", "spectral_entropy", "quantum_advantage", "best_classical"]].mean().round(4)
    print(v.sort_values("effective_rank"))

    # ============ 相关性 ============
    print("\n--- advantage vs 谱判据（Spearman，全数据） ---")
    corr_rows = []
    for key in ["effective_rank", "spectral_entropy", "concentration_index", "gap_ratio", "top5_energy"]:
        rho, p = spearmanr(df[key], df["quantum_advantage"])
        corr_rows.append({"criterion": key, "rho": rho, "p": p})
        print(f"  {key:22s} rho={rho:+.3f} p={p:.4g}")
    pd.DataFrame(corr_rows).to_csv(os.path.join(RESULTS, "exp03_correlation_summary.csv"), index=False)

    # 优势出现的总体情况
    pos = df[df["quantum_advantage"] > 0]
    print(f"\n量子优势>0 的配置: {len(pos)} / {len(df)} ({(100 * len(pos) / len(df)):.1f}%)")
    if len(pos):
        print(f"  优势>0 配置的平均 effective_rank: {pos['effective_rank'].mean():.3f}")
        print(f"  优势<=0 配置的平均 effective_rank: {df[df['quantum_advantage'] <= 0]['effective_rank'].mean():.3f}")

    # ============ 分层相关（防 Simpson 悖论） ============
    print("\n--- 按任务分层：advantage vs effective_rank（Spearman） ---")
    stratified = []
    for t in tasks:
        sub = df[df["task"] == t]
        if len(sub) >= 8:
            rho, p = spearmanr(sub["effective_rank"], sub["quantum_advantage"])
            stratified.append({"task": t, "n": len(sub), "rho": rho, "p": p})
            print(f"  {t:22s} n={len(sub):3d}  rho={rho:+.3f}  p={p:.4g}")
    pd.DataFrame(stratified).to_csv(os.path.join(RESULTS, "exp03_stratified_corr.csv"), index=False)

    # 按任务分层后 rho 的汇总（中位数/均值）
    if stratified:
        rhos = [s["rho"] for s in stratified]
        print(f"  分层 rho 中位数: {np.median(rhos):+.3f}，均值: {np.mean(rhos):+.3f}（与全数据 rho 对比可判断是否存在混杂）")


    # ============ 阈值窗口分析 ============
    print("\n--- 阈值窗口分析（effective_rank 为主判据） ---")
    for crit in ["effective_rank", "spectral_entropy"]:
        bins_df, tau_df = threshold_analysis(df, criterion=crit, n_bins=10)
        print(f"\n[{crit}] 分箱（criterion_med | adv_mean | n | pos_frac）:")
        for _, row in bins_df.iterrows():
            print(f"  {row['criterion_med']:8.3f} | {row['adv_mean']:+.4f} | {int(row['n']):3d} | {row['pos_frac']:.2f}")
        tau_df.to_csv(os.path.join(RESULTS, f"exp03_tau_{crit}.csv"), index=False)
        if len(tau_df):
            sig = tau_df[(tau_df["fdr_p"] < 0.05) if "fdr_p" in tau_df.columns else (tau_df["mannwhitney_p"] < 0.05)]
            if len(sig):
                best = sig.loc[sig["cliff_delta"].idxmax()]
                print(f"  显著阈值: tau={best['tau']:.3f} (fdr_p={best['fdr_p']:.4g}, cliff={best['cliff_delta']:+.3f}, "
                      f"high_mean={best['adv_mean_high']:+.4f} vs low_mean={best['adv_mean_low']:+.4f})")
            else:
                print("  无经 FDR 校正后仍显著的阈值")

    # ============ 可视化 ============
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # 图1：advantage vs effective_rank，按任务着色
    task_colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for i, t in enumerate(tasks):
        sub = df[df["task"] == t]
        axes[0].scatter(sub["effective_rank"], sub["quantum_advantage"],
                        color=task_colors[i], alpha=0.75, s=45, label=t)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Effective rank (spectral criterion)")
    axes[0].set_ylabel("Quantum advantage (acc_Q - best classical)")
    axes[0].set_title("Advantage vs spectral criterion (by task)")
    axes[0].legend(fontsize=8, loc="upper left")

    # 图2：分箱均值曲线（effective_rank）
    bins_df, _ = threshold_analysis(df, criterion="effective_rank", n_bins=10)
    axes[1].errorbar(bins_df["criterion_med"], bins_df["adv_mean"],
                     yerr=bins_df["adv_std"] / np.sqrt(bins_df["n"]),
                     marker="o", capsize=3, color="tab:blue")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Effective rank (bin median)")
    axes[1].set_ylabel("Mean quantum advantage")
    axes[1].set_title("Binned mean advantage: threshold window")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp03_threshold_window.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    # 图3：advantage vs rank 按任务（含正优势分布）——额外图
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    for i, t in enumerate(tasks):
        sub = df[df["task"] == t]
        ax2.scatter(sub["effective_rank"], sub["quantum_advantage"],
                    color=task_colors[i], alpha=0.8, s=45, label=t)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    # 标记正优势区域
    pos_mask = df["quantum_advantage"] > 0
    if pos_mask.any():
        ax2.scatter(df.loc[pos_mask, "effective_rank"], df.loc[pos_mask, "quantum_advantage"],
                    facecolors="none", edgecolors="black", s=120, linewidths=1.2, label="advantage > 0")
    ax2.set_xlabel("Effective rank")
    ax2.set_ylabel("Quantum advantage")
    ax2.set_title("Positive-advantage region highlighted")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    out_fig2 = os.path.join(FIGURES, "exp03_advantage_by_task.png")
    plt.savefig(out_fig2, dpi=150)
    print(f"[保存] {out_fig2}")

    # ============ 结论 ============
    print("\n" + "=" * 70)
    print("结论（exp03 摘要）:")
    print("=" * 70)
    print("  - 优势>0 配置占比、其有效秩均值、以及显著阈值的发现见上方。")
    print("  - 若发现正优势区集中在高有效秩，则 H3 的'谱阈值窗口'得到直接验证。")


if __name__ == "__main__":
    main()
