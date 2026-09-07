"""
tests / test_simulator.py
快速验证自研模拟器的正确性（对比已知解析结果）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from simulator import Simulator, H, X, Z, ry, cnot, cphase


def test_single_qubit_hadamard():
    """|0> 过 H 门 -> (|0>+|1>)/sqrt2。"""
    s = Simulator(1)
    s.apply_1q(H, 0)
    expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    assert np.allclose(s.state, expected), f"got {s.state}, expected {expected}"
    print("PASS: 单比特 Hadamard")


def test_single_qubit_x():
    """|0> 过 X 门 -> |1>。"""
    s = Simulator(1)
    s.apply_1q(X, 0)
    assert np.allclose(s.state, np.array([0, 1], dtype=np.complex128))
    print("PASS: 单比特 X")


def test_twitter_cnot():
    """|10> (qubit0=0? ) 测试 CNOT：控制 qubit0，目标 qubit1。"""
    # 设 n=2, 状态 |q1 q0>。置 qubit0=1 (即 index=1), qubit1=0
    s = Simulator(2)
    s.apply_1q(X, 0)  # |0 1> (qubit0=1)
    s.apply_cnot(control=0, target=1)  # 控制 qubit0, 目标 qubit1 -> |1 1>
    # 期望 index = (q1 q0) = (1,1) = 3
    assert np.allclose(s.state[:4], np.array([0,0,0,1], dtype=np.complex128)), f"got {s.state[:4]}"
    print("PASS: CNOT")


def test_bell_state():
    """制备 (|00>+|11>)/sqrt2，验证纠缠。"""
    s = Simulator(2)
    s.prepare_bell(0, 1)
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    assert np.allclose(s.state, expected), f"got {s.state}, expected {expected}"
    print("PASS: Bell 态制备")


def test_fidelity_same():
    """相同状态保真度应为 1。"""
    s = Simulator(2)
    s.prepare_bell(0, 1)
    assert np.isclose(s.fidelity(s), 1.0)
    print("PASS: 保真度自洽")


def test_fidelity_orthogonal():
    """正交状态保真度应为 0。"""
    s1 = Simulator(1)
    s1.state[:] = np.array([1, 0], dtype=np.complex128)  # |0>
    s2 = Simulator(1)
    s2.state[:] = np.array([0, 1], dtype=np.complex128)  # |1>
    assert np.isclose(s1.fidelity(s2), 0.0)
    print("PASS: 正交保真度")


def test_fidelity_kernel_same_states():
    """一批相同状态，核矩阵应为全 1。"""
    s = Simulator(1)
    states = [s.state.copy() for _ in range(3)]
    K = s.fidelity_kernel(states)
    assert np.allclose(K, np.ones((3, 3)))
    print("PASS: fidelity kernel 全 1")


def test_rz_is_unitary_phase_rotation():
    """R_z 必须是相位旋转（保持模长，|0> 只获得相位因子）。"""
    from simulator import rz
    g = rz(1.0)
    assert np.allclose(g @ g.conj().T, np.eye(2)), "R_z 必须幺正"
    # 作用于 |0>：只改变全局相位（幅度保持 1）
    s = Simulator(1)
    s.apply_1q(g, 0)
    assert np.isclose(np.linalg.norm(s.state), 1.0), "应用 R_z 后状态必须归一"
    # 与 X 基底对比：R_z(π)|+> = e^{-iπ/2}|+> （幅度 1）
    g2 = rz(np.pi)
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    out = g2 @ plus
    assert np.isclose(abs(out[0]), 1 / np.sqrt(2)), "R_z 不得改变幅度"
    print("PASS: R_z 幺正性")


def test_encoding_is_unitary():
    """特征编码后的状态必须归一化（验证编码门链是幺正的）。"""
    from feature_maps import encode_dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4)) * 0.5
    for fm in ["angle", "iqp", "variational", "high_dim"]:
        emb = encode_dataset(X, fm, 5, depth=2)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9), f"{fm}: 编码状态未归一化，norm={norms}"
    print("PASS: 全部编码幺正（状态归一化）")


if __name__ == "__main__":
    test_single_qubit_hadamard()
    test_single_qubit_x()
    test_twitter_cnot()
    test_bell_state()
    test_fidelity_same()
    test_fidelity_orthogonal()
    test_fidelity_kernel_same_states()
    test_rz_is_unitary_phase_rotation()
    test_encoding_is_unitary()
    print("\n=== 全部测试通过 ===")
