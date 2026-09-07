"""
quantum_ml_classicality / src / simulator.py
=============================================
自研量子状态向量模拟器（纯 numpy，无外部量子库依赖）。

设计目标：
- 只依赖 numpy，保证任何机器可复现。
- 面向"小规模"（5-12 qubit）量子特征映射研究。
- 提供与"经典可模拟性"课题匹配的核心原语：
    * 单/双比特门、变分层、数据重载（feature encoding）
    * 量子核矩阵计算（fidelity kernel）
    * 全局测量与简化测量
    * 高效的内积/保真度计算

作者：LightRead 独立课题工作区
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# 常量与单比特门
# ---------------------------------------------------------------------------
# 单比特标准门（2x2），纯复数矩阵
I2 = np.eye(2, dtype=np.complex128)
H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """绕 X 轴旋转门 R_x(theta) = exp(-i theta X / 2)。"""
    c, s = np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    """绕 Y 轴旋转门 R_y(theta)。"""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """绕 Z 轴旋转门 R_z(theta) = diag(e^{-i theta/2}, e^{i theta/2})。"""
    return np.array(
        [[np.exp(-1.0j * theta / 2.0), 0.0], [0.0, np.exp(1.0j * theta / 2.0)]],
        dtype=np.complex128,
    )


def phase(phi: float) -> np.ndarray:
    """相位门 P(phi) = diag(1, e^{i phi})。"""
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * phi)]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# 双比特可控门
# ---------------------------------------------------------------------------
def cnot() -> np.ndarray:
    """受控非门 (CNOT)，4x4，控制位在前 (qubit 约定：高位是 control)。"""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )


def cphase(phi: float) -> np.ndarray:
    """受控相位门 CPhase(phi)，4x4。"""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1.0j * phi)],
        ],
        dtype=np.complex128,
    )


def swap_gate() -> np.ndarray:
    """SWAP 门，4x4。"""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# 张量积辅助
# ---------------------------------------------------------------------------
def _kron_chain(gates: Sequence[np.ndarray]) -> np.ndarray:
    """从左到右对一批矩阵做张量积（qubit 0 在最左=最高位）。"""
    out = gates[0]
    for g in gates[1:]:
        out = np.kron(out, g)
    return out


def _bit_list(n_qubits: int, qubit: int) -> List[int]:
    """构造单比特门作用在 qubit 上的张量积放置：其余位为 I。"""
    # qubit 0 是最低位还是最高位？
    # 我们约定：状态向量 index 的二进制，bit i 对应 qubit i。
    # 即 index = b_0 * 2^0 + b_1 * 2^1 + ...，qubit 0 是最低位。
    # 张量积链从 qubit (n-1) 到 qubit 0 排列（最高位在左）。
    gates = [I2] * n_qubits
    gates[n_qubits - 1 - qubit] = 0  # 占位符，稍后替换
    return gates


# ---------------------------------------------------------------------------
# 状态向量模拟器主类
# ---------------------------------------------------------------------------
class Simulator:
    """
    自研量子状态向量模拟器。

    约定：
    - n_qubits 个 qubit，状态向量维度 dim = 2^n_qubits。
    - 状态向量 |ψ> 排列：索引 i 对应二进制位 b_{n-1}...b_0，
      其中 b_0 是 qubit 0（最低位），b_{n-1} 是 qubit n-1（最高位）。
    - 因此 qubit 0 对应索引的二进制的末位。
    """

    def __init__(self, n_qubits: int, seed: int | None = None):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0  # |0...0>
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # -- 基础操作 ------------------------------------------------------------
    def reset(self) -> None:
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0

    def set_state(self, state: np.ndarray) -> None:
        assert state.shape == (self.dim,), f"expected shape ({self.dim},), got {state.shape}"
        self.state = state.astype(np.complex128).copy()

    def copy(self) -> "Simulator":
        s = Simulator(self.n_qubits)
        s.state = self.state.copy()
        return s

    # -- 应用单比特门 -------------------------------------------------------
    def apply_1q(self, gate: np.ndarray, qubit: int) -> None:
        """在指定 qubit 上作用单比特门 gate（就地修改状态）。"""
        n = self.n_qubits
        # 用 reshape 以 n_qubits 维张量处理：轴 i 对应 qubit (n-1-i)？不对。
        # 让 state reshape 成形状 (2,2,...,2) 共 n 维，索引顺序为
        # [b_{n-1}, b_{n-2}, ..., b_0]，即轴 0 对应 qubit n-1。
        # 那么 qubit q 对应轴 (n-1-q)。
        state_t = self.state.reshape([2] * n)
        axis = n - 1 - qubit
        # np.tensordot 在 axis 上做矩阵乘法
        state_t = np.tensordot(gate, state_t, axes=(1, axis))
        # tensordot 会把 axis 移到最前，需要归位成原始轴序
        order = [axis] + [i for i in range(n) if i != axis]
        state_t = np.transpose(state_t, np.argsort(order))  # 还原轴序
        self.state = state_t.reshape(self.dim)

    # -- 应用双比特门 -------------------------------------------------------
    def apply_2q(self, gate: np.ndarray, qubit0: int, qubit1: int) -> None:
        """
        在 qubit0(控制) 和 qubit1(目标) 上作用 4x4 门。
        gate 的 4x4 索引约定：高位是 qubit0，低位是 qubit1（即 |q0 q1>）。
        """
        if qubit0 == qubit1:
            raise ValueError("qubit0 == qubit1")
        n = self.n_qubits
        state_t = self.state.reshape([2] * n)
        ax0 = n - 1 - qubit0  # 控制位轴
        ax1 = n - 1 - qubit1  # 目标位轴
        # 把 ax0, ax1 移到最前两位
        perm = [ax0, ax1] + [i for i in range(n) if i not in (ax0, ax1)]
        state_t = np.transpose(state_t, perm)
        shape = state_t.shape
        # 合并前两维为 4
        state_t = state_t.reshape(4, -1)
        state_t = gate @ state_t
        # 还原
        state_t = state_t.reshape(shape)
        inv_perm = np.argsort(perm)
        state_t = np.transpose(state_t, inv_perm)
        self.state = state_t.reshape(self.dim)

    def apply_cnot(self, control: int, target: int) -> None:
        self.apply_2q(cnot(), control, target)

    def apply_cphase(self, control: int, target: int, phi: float) -> None:
        self.apply_2q(cphase(phi), control, target)

    # -- 测量 ---------------------------------------------------------------
    def measure_probabilities(self) -> np.ndarray:
        """返回各计算基态的概率分布（实数数组）。"""
        return np.abs(self.state) ** 2

    def measure(self, shots: int = 1000) -> np.ndarray:
        """返回 shots 次采样的计算结果基态索引数组。"""
        probs = self.measure_probabilities()
        return self.rng.choice(self.dim, size=shots, p=probs)

    def expectation(self, op_matrix: np.ndarray, qubits: Sequence[int]) -> complex:
        """在给定 qubit 子集上测量可观察量的期望值。<ψ|O|ψ>。

        op_matrix 作用在 qubits 指定的子空间上，维度应为 2^k。
        （简化：只支持对角可观察量或可通过局域算子的情况）
        """
        # 简化为支持任意局域可观测量：构造整体算符并求期待值
        n = self.n_qubits
        # 构造全空间 POVM 矩阵
        full = self._lift_operator(op_matrix, qubits)
        return np.vdot(self.state, full @ self.state)

    def _lift_operator(self, op: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
        """把作用在 qubits 子集上的矩阵提升为全空间算子。"""
        n = self.n_qubits
        # 构造全空间张量积：非 qubits 位为 I
        # 用逐位 kron
        dim_full = self.dim
        result = np.zeros((dim_full, dim_full), dtype=np.complex128)
        all_qubits = list(range(n))
        free = [q for q in all_qubits if q not in qubits]
        # 对每个自由 bit，遍历
        for basis_out in range(dim_full):
            for basis_in in range(dim_full):
                # 检查 free bits 是否一致
                if any(((basis_out >> q) & 1) != ((basis_in >> q) & 1) for q in free):
                    continue
                # 提取 qubits 子空间的索引对
                sub_out = 0
                sub_in = 0
                for q in qubits:
                    sub_out = (sub_out << 1) | ((basis_out >> q) & 1)
                    sub_in = (sub_in << 1) | ((basis_in >> q) & 1)
                result[basis_out, basis_in] = op[sub_out, sub_in]
        return result

    # -- 量子核 -------------------------------------------------------------
    def fidelity(self, other: "Simulator") -> float:
        """与另一状态的保真度 |<ψ|φ>|^2（若两者都归一化）。"""
        return float(np.abs(np.vdot(self.state, other.state)) ** 2)

    def fidelity_kernel(self, states: Sequence[np.ndarray]) -> np.ndarray:
        """给定一批状态向量（已编码），计算保真度核矩阵。

        注意：此处状态向量应已完成特征编码，且彼此归一化。
        """
        n = len(states)
        # 用矩阵相乘高效计算 |<ψ_i|ψ_j>|^2
        S = np.array(states, dtype=np.complex128)  # shape (n, dim)
        # 内积矩阵 G_ij = <ψ_i|ψ_j>
        G = S @ S.conj().T  # shape (n, n)
        return np.abs(G) ** 2

    # -----------------------------------------------------------------------
    # 便捷高层接口：生成常用初始状态
    # -----------------------------------------------------------------------
    def prepare_bell(self, q0: int, q1: int) -> None:
        """在 q0,q1 上制备 Bell 态 (|00>+|11>)/sqrt2（假定从 |0> 开始）。"""
        self.apply_1q(H, q0)
        self.apply_cnot(q0, q1)


# ---------------------------------------------------------------------------
# 门序列构造辅助（供特征映射使用）
# ---------------------------------------------------------------------------
def build_gate_matrix(gates: Sequence[np.ndarray]) -> np.ndarray:
    """给定一组作用在相同 qubit 上的矩阵，返回乘积矩阵（用于批量）。"""
    result = gates[0]
    for g in gates[1:]:
        result = result @ g
    return result
