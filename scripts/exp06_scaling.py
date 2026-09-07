"""
scripts / exp06_scaling.py
============================
路径1：扩规模实验——8/12/16 qubit 下谱判据、可模拟性与优势关系是否保持。

动机：此前实验最多 5 qubit（32 维状态空间），"经典可模拟性"本质是渐近问题，
审稿人会质疑小规模结论的可外推性。本实验直接在 8/12/16 qubit 上重测核心关系。

设计：
  - 任务: noisy_harmonic（经典难，曾出现正优势）、cross_high（经典易）
  - n_qubits ∈ {8, 12} 全扫描；16 qubit 少量观察配置
  - 映射 ∈ {angle, iqp, high_dim(f15), variational}，depth=2，seeds=3
  - N=250（16 qubit 用 N=150 仅观察）
  - 每配置记录谱判据、RFF 误差、量子优势（对 tuned 经典基线）

输出：results/exp06_scaling.csv + 控制台报告
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from feature_maps import encode_dataset, fidelity_kernel_from_embeddings
from spectral_criteria import compute_all_criteria
from classical_baselines import RandomFourierFeatures, rbf_kernel

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
def make_task(task, n_samples, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, d))
    if task == "noisy_harmonic":
        y = np.sign(np.sin(2 * X[:, 0]) + np.cos(3 * X[:, 1])
                    + 0.5 * X[:, 0] * X[:, 1] + 0.3 * np.sin(4 * X[:, 2]))
        # 20% 标签噪声
        flip = rng.random(n_samples) < 0.2
        y[flip] = -y[flip]
    elif task == "cross_high":
        y = np.sign(X[:, 0] * X[:, 1] + X[:, 2] ** 3 + 0.5 * X[:, 0] * X[:, 2])
    else:
        raise ValueError(task)
    y = np.where(y == 0, 1, y)
    return X, y


def tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed):
    """验证集选 RBF 宽度 + RFF 同宽 + 线性 + 多项式，返回最优精度。"""
    Xtr, Xva, ytr, yva = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
    best_acc = 0.0
    best_gamma = 0.5
    for g in [0.1, 0.5, 2.0, 6.0]:
        Kv = rbf_kernel(Xva, Xtr, gamma=g)
        m = SVC(kernel="precomputed"); m.fit(rbf_kernel(Xtr, Xtr, gamma=g), ytr)
        a = accuracy_score(yva, m.predict(Kv))
        if a > best_acc:
            best_acc, best_gamma = a, g
    # 用最优 gamma 在完整训练集上训练，测试
    K_tr = rbf_kernel(X_tr, gamma=best_gamma)
    m = SVC(kernel="precomputed"); m.fit(K_tr, y_tr)
    acc_rbf = accuracy_score(y_te, m.predict(rbf_kernel(X_te, X_tr, gamma=best_gamma)))
    # RFF
    rff = RandomFourierFeatures(n_features=min(100, len(X_tr)), gamma=best_gamma, seed=seed)
    rff.fit(X_tr)
    m = SVC(kernel="precomputed"); m.fit(rff.kernel(X_tr), y_tr)
    acc_rff = accuracy_score(y_te, m.predict(rff.kernel(X_te, X_tr)))
    # 线性/多项式
    m = SVC(kernel="linear"); m.fit(X_tr, y_tr); acc_lin = accuracy_score(y_te, m.predict(X_te))
    m = SVC(kernel="poly", degree=3); m.fit(X_tr, y_tr); acc_poly = accuracy_score(y_te, m.predict(X_te))
    return max(acc_rbf, acc_rff, acc_lin, acc_poly)


def run_config(task, n_qubits, fm, N, seed, verbose=False):
    d = min(4, n_qubits)  # 特征数 4（与之前一致），qubit 数独立缩放
    t0 = time.time()
    X, y = make_task(task, N, d, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    params = {}
    if fm == "variational":
        params["entangle"] = 1.0
    elif fm == "high_dim":
        params["freq_scale"] = 1.5
    emb = encode_dataset(X_tr, fm, n_qubits, depth=2, **params)
    K = fidelity_kernel_from_embeddings(emb)
    crit = compute_all_criteria(K)

    # 量子 SVM（测试核：用全量 X_tr 的嵌入？为正确需单独编码 X_te）
    emb_te = encode_dataset(X_te, fm, n_qubits, depth=2, **params)
    G = emb_te @ emb.conj().T
    K_te = np.abs(G) ** 2
    m = SVC(kernel="precomputed"); m.fit(K, y_tr)
    acc_q = accuracy_score(y_te, m.predict(K_te))

    best_cl = tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed)

    # RFF 误差（用同一训练数据）
    rff = RandomFourierFeatures(n_features=min(100, len(X_tr)), gamma=0.5, seed=seed)
    rff.fit(X_tr)
    rff_err = rff.approximate_target(K, X_tr)

    if verbose:
        print(f"  [{task}|n={n_qubits}|{fm}|s={seed}] {time.time()-t0:.0f}s "
              f"rank={crit['effective_rank']:.2f} rff={rff_err:.3f} "
              f"adv={acc_q-best_cl:+.3f}")
    return dict(task=task, n_qubits=n_qubits, fm=fm, N=N, seed=seed,
                acc_quantum=acc_q, best_classical=best_cl,
                quantum_advantage=acc_q - best_cl, rff_error=rff_err, **crit)


def main():
    print("=" * 70)
    print("实验 06：扩规模——8/12/16 qubit 下核心关系检验")
    print("=" * 70)
    records = []
    # 8/12 qubit 全扫描
    for task in ["noisy_harmonic", "cross_high"]:
        for n_qubits in [8, 12]:
            for fm in ["angle", "iqp", "variational", "high_dim"]:
                for seed in [0, 1, 2]:
                    records.append(run_config(task, n_qubits, fm, 250, seed, verbose=True))
    # 16 qubit 观察配置（3 映射 × 1 seed）
    for fm in ["angle", "iqp", "high_dim"]:
        records.append(run_config("noisy_harmonic", 16, fm, 150, 0, verbose=True))

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS, "exp06_scaling.csv")
    df.to_csv(out, index=False)
    print(f"\n[保存] {out} ({len(df)} 条)")

    print("\n--- 核心关系随规模变化 ---")
    for n in [8, 12, 16]:
        sub = df[df["n_qubits"] == n]
        r, p = spearmanr(sub["effective_rank"], sub["rff_error"])
        r2, p2 = spearmanr(sub["best_classical"], sub["quantum_advantage"])
        print(f"  n={n:2d} (N={len(sub)}): rank~rff rho={r:+.3f} (p={p:.3f}) | "
              f"adv~classical rho={r2:+.3f} (p={p2:.1e})")

    print("\n--- 谱判据 vs advantage（分层任务）---")
    for task in ["noisy_harmonic", "cross_high"]:
        sub = df[df["task"] == task]
        r, p = spearmanr(sub["effective_rank"], sub["quantum_advantage"])
        print(f"  {task}: rank~adv rho={r:+.3f} p={p:.3f}")
        print(f"    正优势比例: {(sub['quantum_advantage']>0).mean():.1%}  "
              f"平均优势: {sub['quantum_advantage'].mean():+.3f}")

    print("\n--- 平均有效秩 vs qubit 数（concentration 是否随规模加剧）---")
    g = df.groupby(["n_qubits", "fm"])["effective_rank"].mean().round(2)
    print(g)


if __name__ == "__main__":
    main()
