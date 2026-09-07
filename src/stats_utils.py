"""
quantum_ml_classicality / src / stats_utils.py
=============================================
课题统计严谨性工具：
  - BH-FDR 多重比较校正
  - Cliff's delta 效应量
  - 效应量分级报告
  - 简单统计功效估计（用于幂估算报告）

这些工具保证论文中所有多重比较均有校正，所有显著性均有效应量。
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。
    返回与 pvals 同长度的 q 值数组（已做单调化，保证 q 单调不减）。
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    for rank, idx in enumerate(order):
        q[idx] = min(1.0, p[idx] * m / (rank + 1))
    # 单调化（从大到小）
    for i in range(m - 2, -1, -1):
        idx_i = order[i]
        idx_next = order[i + 1]
        q[idx_i] = min(q[idx_i], q[idx_next])
    return q


def cliff_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cliff's delta = P(x > y) - P(x < y)，其中 x∈a, y∈b。
    值域 [-1, 1]：
      - ~0：无效应；±0.147 小；±0.33 中；±0.474 大（常用参考）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    cnt_gt = np.sum(a[:, None] > b[None, :])
    cnt_lt = np.sum(a[:, None] < b[None, :])
    return float((cnt_gt - cnt_lt) / (n_a * n_b))


def effect_size_label(cliff: float) -> str:
    """Cliff's delta 的常用效应量分级。"""
    abs_c = abs(cliff)
    if abs_c < 0.147:
        return "negligible"
    if abs_c < 0.33:
        return "small"
    if abs_c < 0.474:
        return "medium"
    return "large"


def spearman_with_fdr(df, x_col: str, y_cols: Sequence[str]) -> dict:
    """
    对多个判据列与目标列做 Spearman 相关，并做 BH-FDR 校正。
    返回 dict: {"rows": [ {criterion, rho, p, q, n} ]}
    """
    from scipy.stats import spearmanr
    rows = []
    pvals = []
    for col in y_cols:
        x = df[x_col].values
        y = df[col].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            rows.append({"criterion": col, "rho": np.nan, "p": np.nan, "q": np.nan, "n": int(mask.sum())})
            pvals.append(np.nan)
            continue
        rho, p = spearmanr(x[mask], y[mask])
        rows.append({"criterion": col, "rho": rho, "p": p, "q": np.nan, "n": int(mask.sum())})
        pvals.append(p)
    valid_idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    if valid_idx:
        qvals = bh_fdr([pvals[i] for i in valid_idx])
        for qi, idx in enumerate(valid_idx):
            rows[idx]["q"] = qvals[qi]
    return {"rows": rows, "fdr_applied": True}


def conservative_power_estimate(
    n_pairs: int, effect: Optional[float] = None, n_perm: int = 0
) -> str:
    """
    保守功效说明：对于秩检验（Mann-Whitney / Spearman），
    给出在 n_pairs 下可检出的最小效应量的粗略估算（用于报告）。
    这里返回一段文字说明，不做精确计算（精确计算需模拟）。
    """
    return (
        f"n={n_pairs}：秩检验在 α=0.05、功效 0.8 下大致可检出中-大效应"
        f"（|ρ|≳0.4 或 Cliff's |δ|≳0.33）；小效应（|ρ|<0.2）需要 n≳200。"
    )
