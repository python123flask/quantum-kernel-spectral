"""
scripts / exp08_noise.py
==========================
路径2b：噪声实验——退极化通道下谱判据、可模拟性与优势如何变化。

动机：文献表明噪声会改变集中度与可模拟性（Thanasilp 2024、Zendejas 2026），
审稿人需要知道我们的结论在噪声下是否稳健。

设计：
  - 状态编码后用密度矩阵模拟，每层纠缠后插入 depolarizing channel
    ρ -> (1-p)ρ + p I/d  (作用于全部 qubit)
  - n_qubits = 5, N = 100, 任务 = noisy_harmonic
  - 映射 ∈ {angle, iqp, variational} × p ∈ {0, 0.02, 0.1, 0.3} × seeds = 3
  - 混合态保真度核 K_ij = Tr(ρ_i ρ_j)

输出：results/exp08_noise.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from simulator import Simulator, H, X, Y, Z, S, rx, ry, rz, phase, cnot, cphase
from spectral_criteria import compute_all_criteria
from classical_baselines import RandomFourierFeatures, rbf_kernel

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)


def dm_apply_1q(dm, gate, n, qubit):
    """密度矩阵上作用单比特门（轴约定同 simulator）。"""
    idx = [n - 1 - qubit]
    gate_full = np.array([1.0])
    for q in range(n):
        gate_full = np.kron(gate_full, gate if q == idx[0] else np.eye(2))
    return gate_full @ dm @ gate_full.conj().T


def dm_apply_2q(dm, gate, n, q0, q1):
    full = np.array([[1.0 + 0j]])
    for q in range(n):
        if q == n - 1 - q0:
            g = gate  # 4x4：q0 高位、q1 低位
        elif q == n - 1 - q1:
            g = np.eye(2)
        else:
            g = np.eye(2)
        # 需要排列：构造全空间算符
        pass
    # 简化实现：按位构造
    dim = 2 ** n
    full = np.zeros((dim, dim), dtype=np.complex128)
    a, b = n - 1 - q0, n - 1 - q1
    for i in range(dim):
        for j in range(dim):
            ib = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            jb = [(j >> (n - 1 - q)) & 1 for q in range(n)]
            if any(ib[q] != jb[q] for q in range(n) if q not in (a, b)):
                continue
            # 提取 q0,q1 的子索引（q0 高、q1 低）
            si = (ib[a] << 1) | ib[b]
            sj = (jb[a] << 1) | jb[b]
            full[i, j] = gate[si, sj]
    return full @ dm @ full.conj().T


def dm_encoding_noisy(x, n_qubits, depth, p_dep, fm):
    """带退极化噪声的状态编码（密度矩阵）。"""
    dim = 2 ** n_qubits
    dm = np.zeros((dim, dim), dtype=np.complex128)
    dm[0, 0] = 1.0
    d = len(x)
    if fm == "angle":
        for q in range(d):
            dm = dm_apply_1q(dm, ry(x[q]), n_qubits, q)
        return dm
    if fm == "iqp":
        for q in range(n_qubits):
            dm = dm_apply_1q(dm, H, n_qubits, q)
        for q in range(d):
            dm = dm_apply_1q(dm, rz(x[q]), n_qubits, q)
        for layer in range(depth):
            for q in range(d - 1):
                dm = dm_apply_2q(dm, cnot(), n_qubits, q, q + 1)
            for q in range(d):
                dm = dm_apply_1q(dm, phase(x[q] ** 2), n_qubits, q)
            if p_dep > 0:
                dm = (1 - p_dep) * dm + p_dep * np.eye(dim) / dim
        return dm
    if fm == "variational":
        for q in range(d):
            dm = dm_apply_1q(dm, ry(x[q]), n_qubits, q)
        for layer in range(depth):
            for q in range(d - 1):
                dm = dm_apply_2q(dm, cnot(), n_qubits, q, q + 1)
            for q in range(d):
                dm = dm_apply_1q(dm, rz(x[q] * (layer + 1) * 0.5), n_qubits, q)
            if p_dep > 0:
                dm = (1 - p_dep) * dm + p_dep * np.eye(dim) / dim
        return dm
    raise ValueError(fm)


def make_task(n_samples, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, d))
    y = np.sign(np.sin(2 * X[:, 0]) + np.cos(3 * X[:, 1])
                + 0.5 * X[:, 0] * X[:, 1] + 0.3 * np.sin(4 * X[:, 2]))
    flip = rng.random(n_samples) < 0.2
    y[flip] = -y[flip]
    return X, np.where(y == 0, 1, y)


def tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed):
    Xtr, Xva, ytr, yva = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
    best_acc, best_gamma = 0.0, 0.5
    for g in [0.1, 0.5, 2.0, 6.0]:
        m = SVC(kernel="precomputed"); m.fit(rbf_kernel(Xtr, Xtr, gamma=g), ytr)
        a = accuracy_score(yva, m.predict(rbf_kernel(Xva, Xtr, gamma=g)))
        if a > best_acc:
            best_acc, best_gamma = a, g
    m = SVC(kernel="precomputed"); m.fit(rbf_kernel(X_tr, X_tr, gamma=best_gamma), y_tr)
    acc_rbf = accuracy_score(y_te, m.predict(rbf_kernel(X_te, X_tr, gamma=best_gamma)))
    rff = RandomFourierFeatures(n_features=min(80, len(X_tr)), gamma=best_gamma, seed=seed)
    rff.fit(X_tr)
    m = SVC(kernel="precomputed"); m.fit(rff.kernel(X_tr), y_tr)
    acc_rff = accuracy_score(y_te, m.predict(rff.kernel(X_te, X_tr)))
    m = SVC(kernel="linear"); m.fit(X_tr, y_tr); acc_lin = accuracy_score(y_te, m.predict(X_te))
    return max(acc_rbf, acc_rff, acc_lin)


def run_config(fm, p_dep, seed):
    n_q, N, d = 5, 100, 4
    X, y = make_task(N, d, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    # 密度矩阵编码（含噪声）
    dms = [dm_encoding_noisy(x, n_q, 2, p_dep, fm) for x in X_tr]
    K = np.array([[np.real(np.trace(dms[i] @ dms[j])) for j in range(len(dms))] for i in range(len(dms))])
    crit = compute_all_criteria(K)
    dms_te = [dm_encoding_noisy(x, n_q, 2, p_dep, fm) for x in X_te]
    K_te = np.array([[np.real(np.trace(dms_te[i] @ dms[j])) for j in range(len(dms))] for i in range(len(dms_te))])
    m = SVC(kernel="precomputed"); m.fit(K, y_tr)
    acc_q = accuracy_score(y_te, m.predict(K_te))
    best_cl = tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed)
    # RFF 逼近噪声核
    rff = RandomFourierFeatures(n_features=min(80, len(X_tr)), gamma=0.5, seed=seed)
    rff.fit(X_tr)
    rff_err = rff.approximate_target(K, X_tr)
    return dict(fm=fm, p_dep=p_dep, seed=seed, acc_quantum=acc_q, best_classical=best_cl,
                quantum_advantage=acc_q - best_cl, rff_error=rff_err, **crit)


def main():
    print("=" * 70)
    print("实验 08：退极化噪声对谱判据/可模拟性/优势的影响")
    print("=" * 70)
    records = []
    for fm in ["angle", "iqp", "variational"]:
        for p in [0.0, 0.02, 0.1, 0.3]:
            for seed in [0, 1, 2]:
                r = run_config(fm, p, seed)
                print(f"  {fm:12s} p={p:.2f} s={seed} rank={r['effective_rank']:6.2f} "
                      f"rff={r['rff_error']:.3f} adv={r['quantum_advantage']:+.3f}")
                records.append(r)
    df = pd.DataFrame(records)
    out = os.path.join(RESULTS, "exp08_noise.csv")
    df.to_csv(out, index=False)
    print(f"\n[保存] {out} ({len(df)} 条)")
    print("\n--- 噪声强度对判据的影响（平均）---")
    g = df.groupby("p_dep")[["effective_rank", "spectral_entropy", "rff_error",
                             "quantum_advantage"]].mean().round(3)
    print(g)
    r, p = spearmanr(df["p_dep"], df["effective_rank"])
    print(f"\n  p_dep~rank rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(df["p_dep"], df["quantum_advantage"])
    print(f"  p_dep~advantage rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(df["effective_rank"], df["rff_error"])
    print(f"  rank~rff（噪声下）rho={r:+.3f} p={p:.1e}")


if __name__ == "__main__":
    main()
