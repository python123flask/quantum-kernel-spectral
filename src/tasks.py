"""
quantum_ml_classicality / src / tasks.py
=========================================
任务生成器：构造具有可控"难度/结构"的学习任务，
用于研究量子优势何时浮现（谱阈值窗口）。

设计原则：
- 任务必须有真实的非线性/周期/耦合结构，让"经典可胜任但量子核可能有空间"。
- 记录任务的"经典难度"（线性/浅层基线准确率），作为分析中的调节变量。
- 真实数据任务：UCI 风格数据集（PCA 降维到 d 维 + 标准化），保证量子模拟可行。

作者：LightRead 独立课题工作区
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 合成任务族
# ---------------------------------------------------------------------------
def task_harmonic(n: int, d: int, seed: int,
                  n_freq: int = 3, noise_frac: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    谐波结构任务： y = sign( Σ_j a_j sin(2π f_j x_j + φ_j) )。
    特点：数据在频域有结构（量子特征映射天然产生谐波，可能友好）。
    noise_frac: 标签翻转比例（0~1）。
    """
    rng = np.random.default_rng(seed)
    freqs = rng.integers(1, n_freq + 1, size=d).astype(float)
    phases = rng.uniform(0, 2 * np.pi, size=d)
    amps = rng.choice([-1.0, 1.0], size=d) * rng.uniform(0.5, 1.0, size=d)

    X = rng.normal(0, 1.0, size=(n, d))
    raw = np.zeros(n)
    for j in range(d):
        raw += amps[j] * np.sin(2 * np.pi * freqs[j] * X[:, j] + phases[j])
    raw /= np.std(raw) + 1e-9
    y = np.sign(raw)
    y = np.where(y == 0, 1, y)

    if noise_frac > 0:
        flip = rng.random(n) < noise_frac
        y[flip] = -y[flip]
    return X, y


def task_periodic_coupled(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    耦合周期任务：标签由多个变量的周期组合决定（含变量耦合项），
    即 y = sign( cos(2π(x1+x2)) + sin(2π(x1-x2)) + 0.2 sin(2π·3 x3) )。
    这类任务更贴近"数据重载产生交叉项"的量子核能力。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, d))
    raw = (
        np.cos(2 * np.pi * (X[:, 0] + X[:, 1]))
        + np.sin(2 * np.pi * (X[:, 0] - X[:, 1]))
    )
    if d >= 3:
        raw += 0.3 * np.sin(2 * np.pi * 3.0 * X[:, 2])
    if d >= 4:
        raw += 0.2 * np.cos(2 * np.pi * 2.0 * X[:, 3])
    raw /= np.std(raw) + 1e-9
    y = np.sign(raw)
    y = np.where(y == 0, 1, y)
    return X, y


def task_cross_high(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    高阶交叉项任务： y = sign( Σ_{i<j} c_ij x_i x_j + 0.3 Σ x_i^3 )。
    多项式非线性 + 变量交互，经典核（RBF）也可胜任，但考察量子核的"交叉项"能力。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, d))
    raw = np.zeros(n)
    for i in range(d):
        for j in range(i + 1, d):
            raw += rng.uniform(-1, 1) * X[:, i] * X[:, j]
    raw += 0.3 * np.sum(X ** 3, axis=1)
    raw /= np.std(raw) + 1e-9
    y = np.sign(raw)
    y = np.where(y == 0, 1, y)
    return X, y


def task_quantum_friendly(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    "量子友好"任务：标签直接由带不同频率的角度编码组合决定，
    即 y = sign( Σ_j sin(2^{j} π x_j ) )，频率按 1,2,4,8 指数增长。
    这类结构在数学上与 feature map 的傅里叶谱高度重合，
    用于检验"量子核在匹配频域结构时是否更容易显出优势"。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, d))
    raw = np.zeros(n)
    for j in range(d):
        raw += np.sin(2 ** j * np.pi * X[:, j] / 2.0)  # 频率指数增长
    raw /= np.std(raw) + 1e-9
    y = np.sign(raw)
    y = np.where(y == 0, 1, y)
    return X, y


def task_noisy_harmonic(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """带 20% 标签噪声的谐波任务。"""
    return task_harmonic(n, d, seed, n_freq=3, noise_frac=0.2)


# ---------------------------------------------------------------------------
# 真实数据任务
# ---------------------------------------------------------------------------
def task_real_breast_cancer(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Breast Cancer Wisconsin (sklearn 内置)，PCA 降到 d 维后标准化。
    若 sklearn 数据不可用则回退到合成任务。
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X_all = StandardScaler().fit_transform(data.data)
    y_all = data.target.astype(float)
    # 二分类标签换成 ±1：0 -> -1, 1 -> +1
    y_all = 2.0 * y_all - 1.0

    rng = np.random.default_rng(seed)
    pca = PCA(n_components=d).fit(X_all)
    Xp = pca.transform(X_all)
    # 均匀采样（保持类别平衡近似）
    idx = rng.choice(len(Xp), size=n, replace=False)
    return Xp[idx], y_all[idx]


# ---------------------------------------------------------------------------
# 统一注册表
# ---------------------------------------------------------------------------
TASKS: Dict[str, callable] = {
    "harmonic": task_harmonic,
    "periodic_coupled": task_periodic_coupled,
    "cross_high": task_cross_high,
    "quantum_friendly": task_quantum_friendly,
    "noisy_harmonic": task_noisy_harmonic,
    "real_breast_cancer": task_real_breast_cancer,
}


def make_task(name: str, n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """统一任务接口。"""
    fn = TASKS[name]
    return fn(n, d, seed)
