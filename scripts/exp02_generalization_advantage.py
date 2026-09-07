"""
scripts / exp02_generalization_advantage.py
=============================================
课题核心实验：量子 vs 经典泛化优势 与 谱判据的关系（H3）。

流程：
  1. 构造一个"已知存在学习结构"的合成分类任务（ground truth 标签来自某种函数）
  2. 对每个特征映射：编码 → 量子核 → 训练核模型（SVM/KernelRidge）
  3. 同时训练经典核模型（RBF / RFF / 线性）作为对照，测量测试精度
  4. 计算"量子优势" = 量子核测试精度 - 最佳经典核测试精度
  5. 检验"量子优势"是否与谱判据（effective_rank 等）相关 → H3

输出：
  - results/exp02_advantage.csv
  - figures/exp02_advantage_vs_spectrum.png
  - 控制台报告
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score

from feature_maps import encode_dataset, fidelity_kernel_from_embeddings, FEATURE_MAPS
from spectral_criteria import compute_all_criteria, effective_rank
from classical_baselines import rbf_kernel, linear_kernel, polynomial_kernel, RandomFourierFeatures

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def make_classification_dataset(n_samples, d, seed, hard=True):
    """
    构造一个有真实学习结构的二分类任务。
    标签 y = sign( f(x) )，f 是某个非线性函数。
    hard=True 用更复杂的函数（需要更多特征/非线性的任务）。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, d))
    # 非线性标签函数（含交叉项和高频项），模拟"需要非线性核"的任务
    if hard:
        y = (
            np.sign(
                np.sin(2 * X[:, 0])
                + np.cos(3 * X[:, 1])
                + 0.5 * X[:, 0] * X[:, 1]
                + 0.3 * np.sin(4 * X[:, 2]) if d > 2 else np.sin(2 * X[:, 0])
            )
        )
    else:
        y = np.sign(X[:, 0] + 0.5 * X[:, 1])
    # 确保二分类平衡
    y = np.where(y == 0, 1, y)
    return X, y


def random_fourier_kernel(X_train, X_test, n_features, gamma, seed):
    """用 RFF 构造核矩阵（训练+测试）。"""
    rff = RandomFourierFeatures(n_features=n_features, gamma=gamma, seed=seed)
    rff.fit(X_train)
    K_train = rff.kernel(X_train)
    K_test = rff.kernel(X_test, X_train)  # 测试×训练
    return K_train, K_test


def run_single_config(fm_name, n_qubits, depth, X, y, seed):
    """单配置：编码 → 量子核 → 量子SVM → 经典对照 → 计算优势。"""
    # 划分
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    # —— 量子核 ——
    params = {}
    if fm_name == "variational":
        params["entangle"] = 1.0
    elif fm_name == "high_dim":
        params["freq_scale"] = 1.5
    emb_tr = encode_dataset(X_tr, fm_name, n_qubits, depth=depth, **params)
    emb_te = encode_dataset(X_te, fm_name, n_qubits, depth=depth, **params)
    K_q_tr = fidelity_kernel_from_embeddings(emb_tr)
    # 测试×训练核
    G_te = emb_te @ emb_tr.conj().T
    K_q_te = np.abs(G_te) ** 2

    # 量子 SVM
    svm_q = SVC(kernel="precomputed")
    svm_q.fit(K_q_tr, y_tr)
    acc_q = accuracy_score(y_te, svm_q.predict(K_q_te))

    # —— 经典对照 ——
    # 1) RBF
    gamma_rbf = 0.5
    K_rbf_tr = rbf_kernel(X_tr, gamma=gamma_rbf)
    K_rbf_te = rbf_kernel(X_te, X_tr, gamma=gamma_rbf)
    svm_rbf = SVC(kernel="precomputed")
    svm_rbf.fit(K_rbf_tr, y_tr)
    acc_rbf = accuracy_score(y_te, svm_rbf.predict(K_rbf_te))

    # 2) RFF（RBF 的随机特征近似）
    K_rff_tr, K_rff_te = random_fourier_kernel(X_tr, X_te, n_features=min(80, len(X_tr)), gamma=gamma_rbf, seed=seed)
    svm_rff = SVC(kernel="precomputed")
    svm_rff.fit(K_rff_tr, y_tr)
    acc_rff = accuracy_score(y_te, svm_rff.predict(K_rff_te))

    # 3) 线性
    K_l_tr = X_tr @ X_tr.T
    K_l_te = X_te @ X_tr.T
    svm_l = SVC(kernel="precomputed")
    svm_l.fit(K_l_tr, y_tr)
    acc_l = accuracy_score(y_te, svm_l.predict(K_l_te))

    best_classical = max(acc_rbf, acc_rff, acc_l)

    # —— 谱判据（用训练集量子核）——
    crit = compute_all_criteria(K_q_tr)

    return {
        "fm": fm_name,
        "n_qubits": n_qubits,
        "depth": depth,
        "seed": seed,
        "acc_quantum": acc_q,
        "acc_rbf": acc_rbf,
        "acc_rff": acc_rff,
        "acc_linear": acc_l,
        "best_classical": best_classical,
        "quantum_advantage": acc_q - best_classical,
        **crit,
    }


def main():
    print("=" * 70)
    print("实验 02：量子 vs 经典泛化优势 与 谱判据关系（H3）")
    print("=" * 70)

    records = []
    n_samples = 100
    d = 4
    n_qubits = 4  # 用 d 即可，4 qubit
    seeds = [0, 1, 2, 3, 4, 5]

    for fm_name in FEATURE_MAPS:
        for depth in [1, 2, 3]:
            for seed in seeds:
                X, y = make_classification_dataset(n_samples, d, seed, hard=True)
                rec = run_single_config(fm_name, n_qubits, depth, X, y, seed)
                records.append(rec)

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS, "exp02_advantage.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[保存] {out_csv} （{len(df)} 条记录）")

    # === 汇总 ===
    print("\n--- 各特征映射的平均泛化表现 ---")
    gmean = df.groupby("fm")[["acc_quantum", "best_classical", "quantum_advantage", "effective_rank"]].mean().round(4)
    print(gmean)

    # === H3：量子优势 与 谱判据 相关性 ===
    print("\n--- H3：quantum_advantage 与 谱判据的 Spearman 相关 ---")
    criteria_keys = ["effective_rank", "spectral_entropy", "concentration_index", "gap_ratio", "top5_energy"]
    corr_rows = []
    for key in criteria_keys:
        if key not in df.columns:
            continue
        rho, p = spearmanr(df[key], df["quantum_advantage"])
        corr_rows.append({"criterion": key, "spearman_rho": rho, "p_value": p})
        print(f"  {key:22s}  rho={rho:+.3f}   p={p:.4g}")
    pd.DataFrame(corr_rows).to_csv(os.path.join(RESULTS, "exp02_correlation_summary.csv"), index=False)

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"angle": "tab:blue", "iqp": "tab:red", "variational": "tab:green", "high_dim": "tab:orange"}

    # 图1: effective_rank vs quantum_advantage
    for fm, c in colors.items():
        sub = df[df["fm"] == fm]
        axes[0].scatter(sub["effective_rank"], sub["quantum_advantage"], c=c, label=fm, alpha=0.8, s=50)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Effective rank (核谱判据)")
    axes[0].set_ylabel("Quantum advantage (acc_Q - best_classical)")
    axes[0].set_title("H3: 谱判据 vs 量子泛化优势")
    axes[0].legend()

    # 图2: 各特征映射的量子优势箱线 + 谱秩
    fm_order = ["iqp", "angle", "variational", "high_dim"]
    adv_data = [df[df["fm"] == fm]["quantum_advantage"].values for fm in fm_order]
    bp = axes[1].boxplot(adv_data, tick_labels=fm_order)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Quantum advantage")
    axes[1].set_title("量子优势按特征映射分布")
    # 叠加对应有效秩
    for i, fm in enumerate(fm_order):
        axes[1].text(i+1, df[df["fm"]==fm]["quantum_advantage"].max()+0.01,
                     f"rank={df[df['fm']==fm]['effective_rank'].mean():.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp02_advantage_vs_spectrum.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    # === 结论 ===
    print("\n" + "=" * 70)
    print("结论（H3 验证）:")
    print("=" * 70)
    if len(corr_rows) > 0:
        best = max(corr_rows, key=lambda r: abs(r["spearman_rho"]))
        print(f"  最强相关: {best['criterion']} (rho={best['spearman_rho']:+.3f}, p={best['p_value']:.4g})")
    pos = df[df["quantum_advantage"] > 0]
    neg = df[df["quantum_advantage"] <= 0]
    print(f"  量子优势>0 的配置数: {len(pos)} / {len(df)} ({(100*len(pos)/len(df)):.0f}%)")
    print(f"  量子优势>0 组平均有效秩: {pos['effective_rank'].mean():.3f}" if len(pos)>0 else "  无正优势配置")
    print(f"  量子优势<=0 组平均有效秩: {neg['effective_rank'].mean():.3f}" if len(neg)>0 else "  无负优势配置")


if __name__ == "__main__":
    main()
