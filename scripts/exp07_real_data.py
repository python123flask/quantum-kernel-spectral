"""
scripts / exp07_real_data.py
==============================
路径2a：真实数据集实验——sklearn 内置 4 个标准数据集（维度 4-64）。

设计：
  - 数据集: iris(4d), wine(13d), breast_cancer(30d), digits(64d, 二分类 3v8)
  - 预处理: 标准化; wine/breast/digits PCA 到 8 维（与 qubit 对齐）
  - n_qubits = 8, 映射 ∈ {angle, iqp, variational, high_dim}, seeds = 3
  - 指标: 谱判据、RFF 误差、量子优势（tuned 经典基线）

输出：results/exp07_real_data.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from feature_maps import encode_dataset, fidelity_kernel_from_embeddings
from spectral_criteria import compute_all_criteria
from classical_baselines import RandomFourierFeatures, rbf_kernel

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_dataset(name, seed):
    if name == "iris":
        d = load_iris(); X, y = d.data, d.target
        y = (y == 0).astype(int)  # setosa vs rest
    elif name == "wine":
        d = load_wine(); X, y = d.data, d.target
        y = (y == 0).astype(int)
    elif name == "breast_cancer":
        d = load_breast_cancer(); X, y = d.data, d.target
    elif name == "digits":
        d = load_digits(); X, y = d.data, d.target
        mask = np.isin(y, [3, 8])
        X, y = X[mask], (y[mask] == 3).astype(int)
    else:
        raise ValueError(name)
    # 标准化 + PCA 到 min(d, 8)
    X = StandardScaler().fit_transform(X)
    n_comp = min(X.shape[1], 8)
    X = PCA(n_components=n_comp).fit_transform(X)
    return X, y


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
    rff = RandomFourierFeatures(n_features=min(100, len(X_tr)), gamma=best_gamma, seed=seed)
    rff.fit(X_tr)
    m = SVC(kernel="precomputed"); m.fit(rff.kernel(X_tr), y_tr)
    acc_rff = accuracy_score(y_te, m.predict(rff.kernel(X_te, X_tr)))
    m = SVC(kernel="linear"); m.fit(X_tr, y_tr); acc_lin = accuracy_score(y_te, m.predict(X_te))
    m = SVC(kernel="poly", degree=3); m.fit(X_tr, y_tr); acc_poly = accuracy_score(y_te, m.predict(X_te))
    return max(acc_rbf, acc_rff, acc_lin, acc_poly)


def run_config(dataname, fm, seed):
    X, y = load_dataset(dataname, seed)
    n = len(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    n_qubits = X.shape[1]
    params = {}
    if fm == "variational":
        params["entangle"] = 1.0
    elif fm == "high_dim":
        params["freq_scale"] = 1.5
    emb = encode_dataset(X_tr, fm, n_qubits, depth=2, **params)
    K = fidelity_kernel_from_embeddings(emb)
    crit = compute_all_criteria(K)
    emb_te = encode_dataset(X_te, fm, n_qubits, depth=2, **params)
    K_te = np.abs(emb_te @ emb.conj().T) ** 2
    m = SVC(kernel="precomputed"); m.fit(K, y_tr)
    acc_q = accuracy_score(y_te, m.predict(K_te))
    best_cl = tuned_classical_baselines(X_tr, X_te, y_tr, y_te, seed)
    rff = RandomFourierFeatures(n_features=min(100, len(X_tr)), gamma=0.5, seed=seed)
    rff.fit(X_tr)
    rff_err = rff.approximate_target(K, X_tr)
    return dict(dataset=dataname, n_features=X.shape[1], n_samples_total=n, fm=fm, seed=seed,
                acc_quantum=acc_q, best_classical=best_cl, quantum_advantage=acc_q - best_cl,
                rff_error=rff_err, **crit)


def main():
    print("=" * 70)
    print("实验 07：真实数据集（iris/wine/breast_cancer/digits）")
    print("=" * 70)
    records = []
    for dataname in ["iris", "wine", "breast_cancer", "digits"]:
        for fm in ["angle", "iqp", "variational", "high_dim"]:
            for seed in [0, 1, 2]:
                r = run_config(dataname, fm, seed)
                print(f"  {dataname:15s} {fm:12s} s={seed} "
                      f"adv={r['quantum_advantage']:+.3f} rank={r['effective_rank']:7.2f}")
                records.append(r)
    df = pd.DataFrame(records)
    out = os.path.join(RESULTS, "exp07_real_data.csv")
    df.to_csv(out, index=False)
    print(f"\n[保存] {out} ({len(df)} 条)")

    print("\n--- 各数据集量子优势（平均）---")
    g = df.groupby("dataset")[["quantum_advantage", "best_classical", "effective_rank"]].mean().round(3)
    print(g)
    print(f"\n  正优势配置: {(df['quantum_advantage']>0).sum()}/{len(df)} = {(df['quantum_advantage']>0).mean():.1%}")
    r, p = spearmanr(df["best_classical"], df["quantum_advantage"])
    print(f"  adv~classical rho={r:+.3f} p={p:.1e}")
    r, p = spearmanr(df["effective_rank"], df["rff_error"])
    print(f"  rank~rff rho={r:+.3f} p={p:.1e}")


if __name__ == "__main__":
    main()
