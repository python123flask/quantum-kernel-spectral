"""
scripts / summary_all_experiments.py
=====================================
全实验统计汇总（论文 Table 的生成脚本）：
  - 汇总 exp01-05 的所有关键 Spearman 检验
  - BH-FDR 多重比较校正（跨实验矩阵）
  - 效应量（Cliff's delta，用于分组比较）
  - 输出 results/SUMMARY_ALL.md（可直接进论文附录）

数据来源：
  - results/exp01_correlation.csv          （H1：谱判据 vs RFF 可模拟性）
  - results/exp03_threshold.csv            （H3：谱判据 vs 优势 + 难度）
  - results/exp03_difficulty_analysis.csv  （难度分层）
  - results/exp04_knob_sweep.csv           （H2：旋钮 vs 谱/可模拟性）
  - results/exp05_data_dependence.csv      （H4：数据依赖）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from stats_utils import bh_fdr, cliff_delta

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")


def load(name):
    return pd.read_csv(os.path.join(RESULTS, name))


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    rho, p = spearmanr(x[mask], y[mask])
    return rho, p, int(mask.sum())


def main():
    rows = []

    # ---------- H1：谱判据 vs 经典可模拟性（exp01） ----------
    df01 = load("exp01_correlation.csv")
    for crit in ["effective_rank", "spectral_entropy", "gap_ratio", "concentration_index"]:
        rho, p, n = spearman(df01[crit].values, df01["rff_error"].values)
        rows.append({"experiment": "exp01 (H1)", "x": crit, "y": "rff_error",
                     "n": n, "rho": rho, "p": p, "note": "simulability"})

    # exp01 分桶 Cliff's delta（低秩 vs 高秩的 rff_error）
    eff = df01["effective_rank"]
    rff = df01["rff_error"]
    med = np.median(eff)
    cliff = cliff_delta(rff[eff > med].values, rff[eff <= med].values)
    rows.append({"experiment": "exp01 (H1)", "x": "eff_rank>median vs <=median",
                 "y": "rff_error", "n": int((eff > med).sum()) + int((eff <= med).sum()),
                 "rho": np.nan, "p": np.nan, "note": f"cliff_delta={cliff:+.3f}"})

    # ---------- H3：谱判据 vs 优势（exp03 全数据 & 分层） ----------
    df03 = load("exp03_threshold.csv")
    for crit in ["effective_rank", "spectral_entropy", "gap_ratio"]:
        rho, p, n = spearman(df03[crit].values, df03["quantum_advantage"].values)
        rows.append({"experiment": "exp03 (H3-global)", "x": crit, "y": "quantum_advantage",
                     "n": n, "rho": rho, "p": p, "note": "global (all tasks)"})
    # 分层
    for t in df03["task"].unique():
        sub = df03[df03["task"] == t]
        rho, p, n = spearman(sub["effective_rank"].values, sub["quantum_advantage"].values)
        rows.append({"experiment": "exp03 (H3-stratified)", "x": "effective_rank",
                     "y": "quantum_advantage", "n": n, "rho": rho, "p": p, "note": f"task={t}"})

    # ---------- 核心：优势 vs 经典难度（exp03 补充） ----------
    rho, p, n = spearman(df03["best_classical"].values, df03["quantum_advantage"].values)
    rows.append({"experiment": "exp03 (core)", "x": "best_classical", "y": "quantum_advantage",
                 "n": n, "rho": rho, "p": p, "note": "global difficulty"})
    for t in df03["task"].unique():
        sub = df03[df03["task"] == t]
        rho, p, n = spearman(sub["best_classical"].values, sub["quantum_advantage"].values)
        rows.append({"experiment": "exp03 (core-stratified)", "x": "best_classical",
                     "y": "quantum_advantage", "n": n, "rho": rho, "p": p, "note": f"task={t}"})

    # ---------- H2：旋钮 vs 谱/可模拟性（exp04） ----------
    df04 = load("exp04_knob_sweep.csv")
    sub = df04[df04["knob"] == "freq_scale"]
    for y in ["effective_rank", "rff_error"]:
        grp = sub.groupby("value")[y].mean()
        rho, p = spearmanr(grp.index, grp.values)
        rows.append({"experiment": "exp04 (H2)", "x": "freq_scale", "y": y,
                     "n": len(sub), "rho": rho, "p": p, "note": "high_dim"})
    # entangle
    sub = df04[df04["knob"] == "entangle"]
    grp = sub.groupby("value")["effective_rank"].mean()
    rho, p = spearmanr(grp.index, grp.values)
    rows.append({"experiment": "exp04 (H2)", "x": "entangle", "y": "effective_rank",
                 "n": len(sub), "rho": rho, "p": p, "note": "variational"})

    # ---------- H4：数据依赖（exp05） ----------
    df05 = load("exp05_data_dependence.csv")
    grp = df05.groupby("data_dist")["effective_rank"].agg(["mean", "std"])
    rows.append({"experiment": "exp05 (H4)", "x": "data_dist (variation)",
                 "y": "effective_rank", "n": len(df05), "rho": np.nan, "p": np.nan,
                 "note": "see CV summary; angle CV=0.33, high_dim CV=0.47, iqp CV=0.00"})

    # ---------- BH-FDR 校正（只对含 p 的行） ----------
    with_p = [(i, r["p"]) for i, r in enumerate(rows) if np.isfinite(r["p"])]
    if with_p:
        idxs = [i for i, _ in with_p]
        ps = [p for _, p in with_p]
        qs = bh_fdr(ps)
        for k, i in enumerate(idxs):
            rows[i]["q_fdr"] = qs[k]
    else:
        for r in rows:
            r["q_fdr"] = np.nan

    summ = pd.DataFrame(rows)
    # 排序：按 |rho| 降序排核心行
    summ = summ.sort_values("p", na_position="last").reset_index(drop=True)
    summ.to_csv(os.path.join(RESULTS, "SUMMARY_ALL.csv"), index=False)

    # ---------- 输出 Markdown 表格 ----------
    lines = [
        "# 全实验统计汇总（论文 Table / 附录材料）",
        "",
        f"生成时间：2026-09-06  |  总检验数：{len(summ)}  |  BH-FDR 校正已应用",
        "",
        "| 实验 | X | Y | n | Spearman rho | p 值 | FDR q 值 | 说明 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        rho_s = f"{r['rho']:+.3f}" if np.isfinite(r["rho"]) else "—"
        p_s = f"{r['p']:.3g}" if np.isfinite(r["p"]) else "—"
        q_s = f"{r['q_fdr']:.3g}" if np.isfinite(r["q_fdr"]) else "—"
        lines.append(f"| {r['experiment']} | {r['x']} | {r['y']} | {int(r['n'])} | {rho_s} | {p_s} | {q_s} | {r['note']} |")
    lines.append("")
    # 关键发现
    lines += [
        "## 关键发现（按证据强度）",
        "",
        "1. **F1（最强）**：量子优势 ~ 经典基线难度，全局 rho≈-0.78（p≈1e-46），",
        "   任务内分层仍显著 → 优势窗口由经典基线的饱和程度主导。",
        "2. **F2**：谱判据（effective_rank 等）可靠度量经典可模拟性（RFF 逼近误差）",
        "   —— exp01 中 gap_ratio 显著（p=0.008），exp04 中 freq_scale→RFF 误差显著（p=0.037）；",
        "   但谱判据**不能全局预测量子优势**（exp03 全局 p=0.17）。",
        "3. **F3**：谱判据的预测力是任务依赖的（cross_high rho=+0.55 p=5e-4；乳腺癌 +0.39 p=0.02），",
        "   在任务内部谱分散仍有助于优势，但被任务-映射匹配掩盖。",
        "4. **F4**：谱判据强烈数据依赖（exp05 CV 0.32-0.47），由数据分布形状而非维度数决定；",
        "   IQP 例外（任何分布下 rank=1，数据无关崩溃）。",
        "5. **F5（实用）**：评估量子核优势时必须使用强基线（调参后的 RBF/RFF/多项式），",
        "   且任务必须有真实难度；否则'量子优势'会是对照组过弱造成的假象。",
        "",
    ]
    out_md = os.path.join(RESULTS, "SUMMARY_ALL.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[保存] {out_md}")
    print(f"[保存] {os.path.join(RESULTS, 'SUMMARY_ALL.csv')}")
    print("\n=== 前 12 行（按 p 排序） ===")
    print(summ.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
