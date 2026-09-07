"""
quantum_ml_classicality / src / spectral_criteria.py
====================================================
谱判据模块——本课题的核心学术贡献。

目标是构造一系列"高效可计算的谱判据"，能够在训练前预测量子特征映射的
经典可模拟性（是否可被 RFF / Nyström 低秩经典核匹配）。

核心探索的判据（H1/H2/H3 的量化）：
  - effective_rank：有效秩
  - spectral_entropy：谱熵（归一化特征值分布的 Shannon 熵）
  - participation_ratio：参与率（等效自由度）
  - concentration_index：集中度指标（多种）
  - rank_ratio：秩冗余比
  - spectrum_power_law_fit：谱幂律拟合指数（判断谱是否高度集中）
  - gap_ratio：谱隙比（第一与第二特征值比值等）

作者：LightRead 独立课题工作区
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List


def _normalized_spectrum(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对核矩阵 K 做特征分解，返回归一化特征值（降序，非负）作为概率分布。"""
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(np.maximum(eigvals, 0))[::-1]  # 降序，非负
    total = eigvals.sum()
    if total <= eps:
        return np.ones_like(eigvals) / len(eigvals)
    return eigvals / total


def effective_rank(K: np.ndarray) -> float:
    """有效秩 = (Σλ)² / Σλ²，值域 [1, rank]。越接近 1 越集中（越易可模拟）。"""
    eig = np.linalg.eigvalsh(K)
    eig = np.maximum(eig, 0)
    total = eig.sum()
    if total <= 0:
        return 1.0
    return float((total ** 2) / ((eig ** 2).sum() + 1e-30))


def participation_ratio(K: np.ndarray) -> float:
    """参与率 PR = 1 / Σ p_i²，p 为归一化谱。等价于 effective rank（对正定核）。"""
    p = _normalized_spectrum(K)
    return float(1.0 / (np.sum(p ** 2) + 1e-30))


def spectral_entropy(K: np.ndarray) -> float:
    """谱熵（Shannon，以 bit 为单位）H = -Σ p_i log2 p_i。"""
    p = _normalized_spectrum(K)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-30)))


def max_spectral_entropy(n: int) -> float:
    """n 维均匀分布谱的最大熵 = log2(n)。"""
    return float(np.log2(n))


def concentration_index(K: np.ndarray) -> float:
    """
    集中度指数：1 - effective_rank / dim。
    值域 [0, 1)。越接近 1 表示谱越集中（越易经典可模拟）。
    """
    return float(1 - effective_rank(K) / K.shape[0])


def rank_ratio(K: np.ndarray) -> float:
    """秩与维数比值（数值秩）。"""
    n = K.shape[0]
    return float(np.linalg.matrix_rank(K) / n)


def top_k_energy(K: np.ndarray, k: int = 5) -> float:
    """前 k 个特征值占总谱能量的比例（k 可调）。越接近 1 越集中。"""
    if k <= 0:
        return 0.0
    eig = np.linalg.eigvalsh(K)
    eig = np.sort(np.maximum(eig, 0))[::-1]
    total = eig.sum()
    if total <= 0:
        return 0.0
    return float(eig[: min(k, len(eig))].sum() / total)


def gap_ratio(K: np.ndarray) -> float:
    """谱隙比：λ1 / (λ1 + λ2)，衡量头部集中度。"""
    eig = np.linalg.eigvalsh(K)
    eig = np.sort(np.maximum(eig, 0))[::-1]
    if len(eig) < 2:
        return 1.0
    if eig[0] + eig[1] <= 0:
        return 0.0
    return float(eig[0] / (eig[0] + eig[1]))


def power_law_exponent(K: np.ndarray) -> float:
    """
    拟合谱的幂律 λ_k ~ k^(-alpha)，返回 alpha。
    用 log-log 线性回归。alpha 越大谱越集中。
    """
    eig = np.linalg.eigvalsh(K)
    eig = np.sort(np.maximum(eig, 0))[::-1]
    eig = eig[eig > 1e-12]
    if len(eig) < 3:
        return float("nan")
    k = np.arange(1, len(eig) + 1)
    log_k = np.log(k)
    log_eig = np.log(eig)
    # 线性回归斜率
    A = np.vstack([log_k, np.ones_like(log_k)]).T
    slope, _ = np.linalg.lstsq(A, log_eig, rcond=None)[0]
    return float(-slope)


def compute_all_criteria(K: np.ndarray) -> Dict[str, float]:
    """计算全部谱判据，返回字典。"""
    return {
        "effective_rank": effective_rank(K),
        "participation_ratio": participation_ratio(K),
        "spectral_entropy": spectral_entropy(K),
        "concentration_index": concentration_index(K),
        "rank_ratio": rank_ratio(K),
        "top5_energy": top_k_energy(K, k=5),
        "gap_ratio": gap_ratio(K),
        "power_law_exponent": power_law_exponent(K),
    }


# ---------------------------------------------------------------------------
# "组合谱判据"（H3 探索）：把多个判据压缩成一个标量
# ---------------------------------------------------------------------------
def combined_criterion(
    K: np.ndarray,
    w_eff_rank: float = 1.0,
    w_entropy: float = 1.0,
) -> float:
    """
    组合判据 = alpha * 标准化有效秩 + (1-alpha) * 标准化谱熵。
    标准化便于比较。

    具体形式：取 effective_rank / dim 和 spectral_entropy / log2(dim) 的加权平均。
    """
    n = K.shape[0]
    er_norm = effective_rank(K) / n
    ent_norm = spectral_entropy(K) / max(np.log2(n), 1e-9)
    return float(w_eff_rank * er_norm + w_entropy * ent_norm) / (w_eff_rank + w_entropy)
