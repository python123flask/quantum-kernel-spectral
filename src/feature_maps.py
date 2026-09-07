"""
quantum_ml_classicality / src / feature_maps.py
================================================
多种量子特征映射（feature map）的实现。

每种特征映射都把经典数据向量 x ∈ R^d 编码进 n_qubits 个 qubit 的量子态，
并允许通过参数调控"纠缠量 / 数据重载深度"，用于系统扫描谱结构。

核心接口：
    encode(x, n_qubits, params) -> state_vector (numpy complex array)
    kernel_matrix(dataset, feature_map, params) -> 保真度核矩阵

作者：LightRead 独立课题工作区
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence

from simulator import Simulator, H, X, Y, Z, S, rx, ry, rz, phase, cnot, cphase


# ---------------------------------------------------------------------------
# 特征映射族 1：角度编码（Angle Encoding）
# ---------------------------------------------------------------------------
def angle_encoding(x: np.ndarray, n_qubits: int, depth: int = 1) -> np.ndarray:
    """
    角度编码：把 x 的每一维映射为一个 qubit 的 R_y 旋转角。

    x 长度必须 <= n_qubits。
    这是"最朴素"的编码——量子核通常等价于一个简单的核（高相关、易集中）。
    """
    d = len(x)
    assert d <= n_qubits, f"x dim {d} > n_qubits {n_qubits}"
    s = Simulator(n_qubits)
    for q in range(d):
        s.apply_1q(ry(x[q]), q)
    return s.state


# ---------------------------------------------------------------------------
# 特征映射族 2：IQP 编码（Instantaneous Quantum Polynomial）
# ---------------------------------------------------------------------------
def iqp_encoding(x: np.ndarray, n_qubits: int, depth: int = 1) -> np.ndarray:
    """
    IQP 编码（Havlíček et al. 变体）：
      1) H 层（|0> → 叠加态，使数据门产生非平凡相位）
      2) 每个 qubit 做 R_z 旋转，角由 x 的线性项决定
      3) 交替 CNOT 阶梯 + 二次相位 + 数据重载
    这是经典可模拟性文献中最常用的编码之一。
    """
    d = len(x)
    assert d <= n_qubits
    s = Simulator(n_qubits)
    # H 层：产生叠加态（IQP 标准构造）
    for q in range(n_qubits):
        s.apply_1q(H, q)
    # 第一层：R_z 编码
    for q in range(d):
        s.apply_1q(rz(x[q]), q)
    # 纠缠层 + 二次相位 + 数据重载
    for layer in range(depth):
        # CNOT 阶梯
        for q in range(d - 1):
            s.apply_cnot(q, q + 1)
        # 二次相位项
        for q in range(d):
            s.apply_1q(phase(x[q] ** 2), q)
        # 数据重载（增量相位）
        if layer < depth - 1:
            for q in range(d):
                s.apply_1q(rz(x[q] * 0.5), q)
    return s.state


# ---------------------------------------------------------------------------
# 特征映射族 3：变分特征映射（Havlíček / Schuld 型）
# ---------------------------------------------------------------------------
def variational_encoding(
    x: np.ndarray, n_qubits: int, depth: int = 1, entangle: float = 1.0
) -> np.ndarray:
    """
    变分特征映射：
      - 先角度编码
      - 然后若干层 (数据重载 + 纠缠层)
      entangle 控制在纠缠层的强度。

    这是"真实"QML 常用的结构，其谱行为更复杂、更有研究价值。
    """
    d = len(x)
    assert d <= n_qubits
    s = Simulator(n_qubits)
    # 初始角度编码
    for q in range(d):
        s.apply_1q(ry(x[q]), q)
    for layer in range(depth):
        # 纠缠层（CNOT 阶梯，强度可调）
        for q in range(d - 1):
            if entangle >= 1.0:
                s.apply_cnot(q, q + 1)
            else:
                # 弱纠缠：用受控相位替代（相位幅度可控）
                s.apply_cphase(q, q + 1, entangle * np.pi)
        # 数据重载（再次编码）
        for q in range(d):
            s.apply_1q(rz(x[q] * (layer + 1) * 0.5), q)
    return s.state


# ---------------------------------------------------------------------------
# 特征映射族 4：高维/纠缠增强映射（用于构造"有优势"的谱）
# ---------------------------------------------------------------------------
def high_dim_encoding(
    x: np.ndarray, n_qubits: int, depth: int = 1, freq_scale: float = 1.0
) -> np.ndarray:
    """
    高维纠缠增强编码：
      - 每个数据维映射到多个 qubit（通过不同频率调制）
      - 大频率尺度 freq_scale 控制傅里叶谱扩散（影响集中度）
    这是构造"谱结构可调"的最直接手段。
    """
    d = len(x)
    # 这里用重复采样：把 x 扩到 n_qubits 维
    reps = n_qubits
    s = Simulator(reps)
    # 第一层：角度编码（用不同频率）
    for q in range(reps):
        freq = freq_scale * (q + 1)
        # 用 (q % d) 对应的 x 分量，乘不同频率
        xc = x[q % d]
        s.apply_1q(ry(xc * freq), q)
        s.apply_1q(rz(xc * freq * 0.7), q)
    # 纠缠层
    for layer in range(depth):
        for q in range(reps - 1):
            s.apply_cphase(q, q + 1, np.pi / 2)
    return s.state


# ---------------------------------------------------------------------------
# 特征映射统一接口
# ---------------------------------------------------------------------------
FEATURE_MAPS: Dict[str, Callable] = {
    "angle": angle_encoding,
    "iqp": iqp_encoding,
    "variational": variational_encoding,
    "high_dim": high_dim_encoding,
}


def encode_dataset(
    X: np.ndarray,
    feature_map: str,
    n_qubits: int,
    depth: int = 1,
    entangle: float = 1.0,
    freq_scale: float = 1.0,
) -> np.ndarray:
    """
    对数据集 X (n_samples, d) 逐行用指定特征映射编码，返回状态矩阵 (n_samples, dim)。

    返回状态矩阵第 i 行是第 i 个样本编码后的量子态。
    """
    n_samples = X.shape[0]
    dim = 2 ** n_qubits
    emb = np.zeros((n_samples, dim), dtype=np.complex128)
    fm = FEATURE_MAPS[feature_map]
    for i in range(n_samples):
        x = X[i]
        if feature_map == "variational":
            emb[i] = fm(x, n_qubits, depth, entangle)
        elif feature_map == "high_dim":
            emb[i] = fm(x, n_qubits, depth, freq_scale)
        else:
            emb[i] = fm(x, n_qubits, depth)
    return emb


# ---------------------------------------------------------------------------
# 保真度核矩阵计算
# ---------------------------------------------------------------------------
def fidelity_kernel_from_embeddings(emb: np.ndarray) -> np.ndarray:
    """
    给定编码后的状态矩阵 (n, dim)，计算保真度核 K_ij = |<ψ_i|ψ_j>|^2。
    高效实现：利用矩阵乘法。
    """
    # emb shape (n, dim)
    G = emb @ emb.conj().T  # (n, n) 复内积
    return np.abs(G) ** 2


# ---------------------------------------------------------------------------
# 特征映射的"傅里叶谱"分析工具（供谱判据用）
# ---------------------------------------------------------------------------
def kernel_fourier_spectrum(kernel_matrix: np.ndarray) -> np.ndarray:
    """
    对核矩阵做特征分解，返回特征值（降序）。
    用于计算谱判据。
    """
    eigvals = np.linalg.eigvalsh(kernel_matrix)
    return np.sort(np.maximum(eigvals, 0))[::-1]


def effective_rank(kernel_matrix: np.ndarray) -> float:
    """
    有效秩 effective rank = (Σλ)² / Σλ²  （对核矩阵特征值）。
    对正定核：值域 [1, rank]，越接近 1 越集中。
    """
    eig = kernel_fourier_spectrum(kernel_matrix)
    total = eig.sum()
    if total <= 0:
        return 1.0
    return float((total ** 2) / (eig ** 2).sum())


if __name__ == "__main__":
    # 自测：编码几种特征映射，看有效秩差异
    rng = np.random.default_rng(0)
    n_samples, n_qubits = 20, 4
    X = rng.normal(size=(n_samples, 3)) * 0.5
    for name in FEATURE_MAPS:
        emb = encode_dataset(X, name, n_qubits, depth=2)
        K = fidelity_kernel_from_embeddings(emb)
        r = effective_rank(K)
        print(f"{name:15s} rank={np.linalg.matrix_rank(K):2d} eff_rank={r:.4f}")
