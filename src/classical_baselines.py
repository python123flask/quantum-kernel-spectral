"""
quantum_ml_classicality / src / classical_baselines.py
======================================================
经典对照模型——用于判定量子特征映射的"经典可模拟性"。

核心思想：给定量子核 K_Q（由量子特征映射诱导），
我们构造一系列"经典候选核"，并测量经典核能否在泛化能力上匹配/超越量子核。

包括：
- Random Fourier Features (RFF) 核近似
- Nyström 低秩近似
- 经典核（RBF / 多项式 / 线性）
- "匹配谱"的经典模型

作者：LightRead 独立课题工作区
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------------
# Random Fourier Features (RFF)
# ---------------------------------------------------------------------------
class RandomFourierFeatures:
    """
    Random Fourier Features（Rahimi & Recht 2007）。

    用随机 Fourier 特征 z(x) 来逼近一个（近）平移不变核 k(x,y) ≈ z(x)·z(y)。
    当量子核能被少量 RFF 逼近时，说明对应的量子特征映射可被经典高效模拟。
    """

    def __init__(self, n_features: int, gamma: float = 1.0, seed: int = 0):
        self.n_features = n_features
        self.gamma = gamma
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._W = None   # 随机频率矩阵 (n_features, d)
        self._b = None   # 随机相位 (n_features,)

    def fit(self, X: np.ndarray) -> "RandomFourierFeatures":
        d = X.shape[1]
        # RBF 核 k(x,y)=exp(-gamma ||x-y||^2) 对应的 Fourier 采样
        self._W = self._rng.normal(0, np.sqrt(2 * self.gamma), size=(self.n_features, d))
        self._b = self._rng.uniform(0, 2 * np.pi, size=self.n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._W is not None, "must fit first"
        # z(x) = sqrt(1/n_features) * [cos(Wx+b), sin(Wx+b)]  (或仅 cos)
        proj = X @ self._W.T + self._b  # (n, n_features)
        return np.cos(proj) * np.sqrt(2.0 / self.n_features)

    def kernel(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """计算 RFF 近似的核矩阵 K = z(X) @ z(Y)^T。"""
        Zx = self.transform(X)
        if Y is None:
            return Zx @ Zx.T
        Zy = self.transform(Y)
        return Zx @ Zy.T

    def approximate_target(self, K_target: np.ndarray, X: np.ndarray) -> float:
        """
        用 RFF 核逼近目标核 K_target（训练数据上）。
        返回相对误差 ||K_rff - K_target||_F / ||K_target||_F。
        """
        K_rff = self.kernel(X)
        norm_target = np.linalg.norm(K_target)
        if norm_target == 0:
            return 0.0
        return float(np.linalg.norm(K_rff - K_target) / norm_target)


# ---------------------------------------------------------------------------
# Nyström 低秩近似
# ---------------------------------------------------------------------------
def nystrom_approximation(K: np.ndarray, n_landmark: int = 20, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nyström 方法：从 K 中选 n_landmark 行/列，构造低秩近似 K ≈ C W^+ C^T。

    返回 (K_approx, error)。
    """
    n = K.shape[0]
    n_landmark = min(n_landmark, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_landmark, replace=False)
    C = K[:, idx]     # (n, m)
    W = K[np.ix_(idx, idx)]  # (m, m)
    # W 伪逆
    W_inv = np.linalg.pinv(W)
    K_approx = C @ W_inv @ C.T
    error = np.linalg.norm(K_approx - K) / (np.linalg.norm(K) + 1e-12)
    return K_approx, error


# ---------------------------------------------------------------------------
# 经典核族
# ---------------------------------------------------------------------------
def rbf_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = 1.0) -> np.ndarray:
    if Y is None:
        Y = X
    sq = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq, 0))


def linear_kernel(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    if Y is None:
        Y = X
    return X @ Y.T


def polynomial_kernel(X: np.ndarray, Y: np.ndarray = None, degree: int = 2, gamma: float = 1.0) -> np.ndarray:
    if Y is None:
        Y = X
    return (gamma * (X @ Y.T) + 1) ** degree


# ---------------------------------------------------------------------------
# "匹配谱"的经典模型（把量子核的特征值谱复制给经典核）
# ---------------------------------------------------------------------------
def matched_spectrum_classical_kernel(K_quantum: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    构造一个"谱匹配"的经典核：
      取 K_quantum 的特征分解 K = U Λ U^T，
      用"随机的正交基但相同的特征值谱"构造 K_classic = V Λ V^T。

    如果 K_classic 与 K_quantum 的泛化性能接近，说明量子核没有用上比"特征值谱"
    更丰富的信息（即可被低秩经典核匹配）。
    """
    n = K_quantum.shape[0]
    eigvals, U = np.linalg.eigh(K_quantum)
    # 保特征值，换随机正交基
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    # 对齐方向（保持正定性）
    K_classic = Q @ np.diag(np.maximum(eigvals, 0)) @ Q.T
    return K_classic


# ---------------------------------------------------------------------------
# 经典可模拟性判定：统一入口
# ---------------------------------------------------------------------------
def classical_simulability_metrics(
    K_quantum: np.ndarray,
    X: np.ndarray,
    rff_n_features: int = 60,
    rff_gamma: float = 1.0,
    nystrom_landmarks: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """
    给定量子核 K_quantum 和原始数据 X，计算一组"经典可模拟性"度量。

    返回 dict，包含：
      - rff_error: RFF 逼近视误差（越小越可模拟）
      - nystrom_error: Nyström 低秩近似误差
      - eff_rank: 有效秩
      - spectral_entropy: 谱熵
      - rank_ratio: 秩 / 维度 比例
    """
    metrics = {}
    # RFF 逼近
    d = X.shape[1]
    rff = RandomFourierFeatures(n_features=rff_n_features, gamma=rff_gamma, seed=seed)
    rff.fit(X)
    metrics["rff_error"] = rff.approximate_target(K_quantum, X)

    # Nyström
    _, nys_err = nystrom_approximation(K_quantum, n_landmark=nystrom_landmarks, seed=seed)
    metrics["nystrom_error"] = nys_err

    # 谱度量
    from spectral_criteria import effective_rank, spectral_entropy, participation_ratio
    metrics["eff_rank"] = effective_rank(K_quantum)
    metrics["spectral_entropy"] = spectral_entropy(K_quantum)
    metrics["participation_ratio"] = participation_ratio(K_quantum)
    metrics["rank_ratio"] = np.linalg.matrix_rank(K_quantum) / K_quantum.shape[0]
    return metrics


# ---------------------------------------------------------------------------
# 泛化优势测量：训练经典模型，测量量子 vs 经典的性能差距
# ---------------------------------------------------------------------------
def train_predict_common(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel_type: str = "kernel_ridge",
    alpha: float = 1e-3,
    **kwargs,
) -> Dict[str, float]:
    """
    在给定核矩阵上训练一个核模型，测量测试性能。
    用于对齐比较：同一样本上用"量子核"和"经典核"分别训练，对比性能。
    """
    if kernel_type == "kernel_ridge":
        # KernelRidge 需要传入预计算核，这里直接构造
        model = KernelRidge(alpha=alpha, kernel="precomputed")
        model.fit(K_train, y_train)
        pred = model.predict(K_test)
        perf = {"mse": float(mean_squared_error(y_test, pred)),
                "acc": float(accuracy_score(np.sign(y_test), np.sign(pred))) if len(np.unique(y_test)) > 2 else None}
    elif kernel_type == "svc":
        model = SVC(kernel="precomputed")
        model.fit(K_train, y_train)
        pred = model.predict(K_test)
        perf = {"mse": None, "acc": float(accuracy_score(y_test, pred))}
    else:
        raise ValueError(f"unknown kernel_type {kernel_type}")
    return perf
