"""
scripts / exp11_amp_damping.py
================================
路径2补充：振幅阻尼（amplitude damping）噪声下的稳健性。

exp08 只测试了退极化通道。本实验加入振幅阻尼：每层纠缠后对每个 qubit
施加 AD channel: ρ -> K1 ρ K1† + K2 ρ K2†
  K1 = diag(1, sqrt(1-γ)),  K2 = [[0, sqrt(γ)],[0,0]]
  γ ∈ {0, 0.05, 0.2, 0.4},  n_qubits=5, N=100
输出：results/exp11_amp_damping.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from simulator import H, X, Y, Z, ry, rz, phase, cnot
from classical_baselines import RandomFourierFeatures, rbf_kernel

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)


def dm_apply_1q(dm, gate, n, qubit):
    dim = 2 ** n
    full = np.zeros((dim, dim), dtype=np.complex128)
    gate = gate.astype(np.complex128)
    for i in range(dim):
        for j in range(dim):
            ib = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            jb = [(j >> (n - 1 - q)) & 1 for q in range(n)]
            if any(ib[q] != jb[q] for q in range(n) if q != qubit):
                continue
            full[i, j] = gate[ib[qubit], jb[qubit]]
    return full @ dm @ full.conj().T


def dm_apply_2q(dm, gate, n, q0, q1):
    dim = 2 ** n
    full = np.zeros((dim, dim), dtype=np.complex128)
    gate = gate.astype(np.complex128)
    a, b = q0, q1
    for i in range(dim):
        for j in range(dim):
            ib = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            jb = [(j >> (n - 1 - q)) & 1 for q in range(n)]
            if any(ib[q] != jb[q] for q in range(n) if q not in (a, b)):
                continue
            si = (ib[a] << 1) | ib[b]
            sj = (jb[a] << 1) | jb[b]
            full[i, j] = gate[si, sj]
    return full @ dm @ full.conj().T


def amp_damping_channel(dm, n, gamma):
    """对每个 qubit 施加振幅阻尼（作用后重新归一不需要，Kraus 保持 trace）。"""
    dim = 2 ** n
    K1 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K2 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    out = np.zeros_like(dm)
    for q in range(n):
        # 在 qubit q 上作用 K1/K2 的 lifting（直接从 dm 计算，避免多次 full 矩阵）
        pass
    # 简单实现：逐 qubit 用 lifting
    for q in range(n):
        # 作用 K1@rho@K1† + K2@rho@K2† 到该 qubit
        m1 = lift_1q(n, q, K1)
        m2 = lift_1q(n, q, K2)
        dm = m1 @ dm @ m1.conj().T + m2 @ dm @ m2.conj().T
    return dm


def lift_1q(n, qubit, g):
    dim = 2 ** n
    G = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            ib = (i >> (n - 1 - qubit)) & 1
            jb = (j >> (n - 1 - qubit)) & 1
            if all((i >> (n - 1 - q)) & 1 == (j >> (n - 1 - q)) & 1
                   for q in range(n) if q != qubit):
                G[i, j] = g[ib, jb]
    return G


def dm_encode_ad(x, n_qubits, depth, gamma, fm):
    dim = 2 ** n_qubits
    dm = np.zeros((dim, dim), dtype=np.complex128)
    dm[0, 0] = 1.0
    d = len(x)
    if fm == "angle":
        for q in range(d):
            dm = dm_apply_1q(dm, ry(x[q]), n_qubits, q)
        if gamma > 0:
            dm = amp_damping_channel(dm, n_qubits, gamma)
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
            if gamma > 0:
                dm = amp_damping_channel(dm, n_qubits, gamma)
        return dm
    if fm == "variational":
        for q in range(d):
            dm = dm_apply_1q(dm, ry(x[q]), n_qubits, q)
        for layer in range(depth):
            for q in range(d - 1):
                dm = dm_apply_2q(dm, cnot(), n_qubits, q, q + 1)
            for q in range(d):
                dm = dm_apply_1q(dm, rz(x[q] * (layer + 1) * 0.5), n_qubits, q)
            if gamma > 0:
                dm = amp_damping_channel(dm, n_qubits, gamma)
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


def run_config(fm, gamma, seed):
    n_q, N, d = 5, 100, 4
    X, y = make_task(N, d, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    dms = [dm_encode_ad(x, n_q, 2, gamma, fm) for x in X_tr]
    K = np.array([[np.real(np.trace(dms[i] @ dms[j])) for j in range(len(dms))] for i in range(len(dms))])
    from spectral_criteria import compute_all_criteria
    crit = compute_all_criteria(K)
    dms_te = [dm_encode_ad(x, n_q, 2, gamma, fm) for x in X_te]
    K_te = np.array([[np.real(np.trace(dms_te[i] @ dms[j])) for j in range(len(dms))] for i in range(len(dms_te))])
    m = SVC(kernel="precomputed"); m.fit(K, y_tr)
    acc_q = accuracy_score(y_te, m.predict(K_te))
    best_cl = tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed)
    rff = RandomFourierFeatures(n_features=min(80, len(X_tr)), gamma=0.5, seed=seed)
    rff.fit(X_tr)
    rff_err = rff.approximate_target(K, X_tr)
    return dict(fm=fm, gamma=gamma, seed=seed, acc_quantum=acc_q, best_classical=best_cl,
                quantum_advantage=acc_q - best_cl, rff_error=rff_err, **crit)


def main():
    print("=" * 70)
    print("实验 11：振幅阻尼噪声稳健性")
    print("=" * 70)
    records = []
    for fm in ["angle", "iqp", "variational"]:
        for gamma in [0.0, 0.05, 0.2, 0.4]:
            for seed in [0, 1, 2]:
                r = run_config(fm, gamma, seed)
                print(f"  {fm:12s} g={gamma:.2f} s={seed} rank={r['effective_rank']:6.2f} "
                      f"rff={r['rff_error']:.3f} adv={r['quantum_advantage']:+.3f}")
                records.append(r)
    df = pd.DataFrame(records)
    out = os.path.join(RESULTS, "exp11_amp_damping.csv")
    df.to_csv(out, index=False)
    print(f"\n[保存] {out} ({len(df)} 条)")
    g = df.groupby("gamma")[["effective_rank", "spectral_entropy", "rff_error",
                             "quantum_advantage"]].mean().round(3)
    print("\n--- 振幅阻尼强度对判据的影响（平均）---")
    print(g)
    r, p = spearmanr(df["gamma"], df["effective_rank"])
    print(f"\n  gamma~rank rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(df["gamma"], df["quantum_advantage"])
    print(f"  gamma~advantage rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(df["effective_rank"], df["rff_error"])
    print(f"  rank~rff（AD下）rho={r:+.3f} p={p:.1e}")


if __name__ == "__main__":
    main()
