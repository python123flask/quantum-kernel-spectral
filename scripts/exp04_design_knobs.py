"""
scripts / exp04_design_knobs.py
================================
补充实验：设计旋钮 → 谱结构 / 可模拟性映射（H2 相关）。

研究问题：量子特征映射的哪些"设计旋钮"（深度、纠缠强度、频率尺度）
能有效改变核矩阵的谱结构（分散 vs 集中）与经典可模拟性（RFF 逼近难度）？

若 H2 成立：频率尺度/纠缠等旋钮改变傅里叶谱的集中度 ⇒ 核谱集中度变化 ⇒
可模拟性变化。本实验给出"如何构造有优势潜力的核"的实用映射表。

输出：
  - results/exp04_knob_sweep.csv
  - figures/exp04_knob_effects.png（旋钮→有效秩/可模拟性曲线）
  - figures/exp04_freq_vs_rank.png（频率尺度→谱结构基线）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from tasks import make_task
from feature_maps import encode_dataset, fidelity_kernel_from_embeddings
from spectral_criteria import compute_all_criteria, effective_rank, spectral_entropy
from classical_baselines import RandomFourierFeatures

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def compute_metrics(fm, kwargs, X, n_qubits, seed):
    """编码 → 核 → 谱判据 + RFF 逼近误差。"""
    emb = encode_dataset(X, fm, n_qubits, **kwargs)
    K = fidelity_kernel_from_embeddings(emb)
    crit = compute_all_criteria(K)
    rff = RandomFourierFeatures(n_features=min(80, len(X)), gamma=0.5, seed=seed)
    rff.fit(X)
    crit["rff_error"] = rff.approximate_target(K, X)
    return crit


def main():
    print("=" * 70)
    print("实验 04：设计旋钮 → 谱结构 / 可模拟性映射（H2）")
    print("=" * 70)

    n_samples = 80
    d = 4
    n_qubits = 5
    X, _ = make_task("harmonic", n_samples, d, seed=0)
    X = StandardScaler().fit_transform(X)
    seeds = [0, 1, 2]

    records = []

    # --- 1) high_dim: freq_scale 扫描 ---
    print("\n[1] high_dim: freq_scale ∈ {0.5, 1, 2, 4, 8} × depth ∈ {1,2,3}")
    for freq in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for depth in [1, 2, 3]:
            for seed in seeds:
                rec = compute_metrics("high_dim", {"depth": depth, "freq_scale": freq}, X, n_qubits, seed)
                records.append({"knob": "freq_scale", "fm_label": "high_dim", "value": freq, "depth": depth, "seed": seed, **rec})

    # --- 2) variational: entangle 扫描 ---
    print("[2] variational: entangle ∈ {0, 0.2, 0.5, 0.8, 1.0} × depth ∈ {1,2,3}")
    for ent in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for depth in [1, 2, 3]:
            for seed in seeds:
                rec = compute_metrics("variational", {"depth": depth, "entangle": ent}, X, n_qubits, seed)
                records.append({"knob": "entangle", "fm_label": "variational", "value": ent, "depth": depth, "seed": seed, **rec})

    # --- 3) iqp: depth 扫描 ---
    print("[3] iqp: depth ∈ {1,2,3,4,5}")
    for depth in [1, 2, 3, 4, 5]:
        for seed in seeds:
            rec = compute_metrics("iqp", {"depth": depth}, X, n_qubits, seed)
            records.append({"knob": "depth", "fm_label": "iqp", "value": depth, "depth": depth, "seed": seed, **rec})

    # --- 4) angle: depth 扫描 ---
    print("[4] angle: depth ∈ {1,2,3}")
    for depth in [1, 2, 3]:
        for seed in seeds:
            rec = compute_metrics("angle", {"depth": depth}, X, n_qubits, seed)
            records.append({"knob": "depth", "fm_label": "angle", "value": depth, "depth": depth, "seed": seed, **rec})

    df = pd.DataFrame(records)
    # 标注 feature_map（修复：depth 旋钮区分 angle/iqp）
    df["feature_map"] = df["fm_label"]

    out_csv = os.path.join(RESULTS, "exp04_knob_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[保存] {out_csv} （{len(df)} 条记录）")

    # 关键：分析基于“写出的 CSV 重新读回”的副本，
    # 保证论文/脚本输出与公开数据（CSV）可用数值完全一致
    # （内存浮点值与十进制写回在近并列处可能产生秩次差异）
    df = pd.read_csv(out_csv)

    # ============ 分析 ============
    print("\n--- 旋钮与谱结构的相关性（按旋钮分组） ---")
    summary_rows = []
    for knob in ["freq_scale", "entangle", "depth"]:
        sub = df[df["knob"] == knob]
        # 对 depth 旋钮：区分 angle 与 iqp
        if knob == "depth":
            for fm in ["angle", "iqp"]:
                sub2 = sub[sub["feature_map"] == fm].sort_values("depth")
                if len(sub2) < 6:
                    continue
                # 全配置口径相关（避免小样本均值相关的渐近 p 值失效：
                # scipy spearmanr 在 n=5 完全单调时 p 值不可靠）
                rho_r, p_r = spearmanr(sub2["value"], sub2["effective_rank"])
                rho_e, p_e = spearmanr(sub2["value"], sub2["rff_error"])
                summary_rows.append({"knob": knob, "fm": fm, "n": len(sub2),
                                     "rho_rank_vs_knob": rho_r, "p_rank": p_r,
                                     "rho_rff_vs_knob": rho_e, "p_rff": p_e})
                print(f"  depth({fm}): rank vs depth rho={rho_r:+.3f} (p={p_r:.3g}) | rff vs depth rho={rho_e:+.3f} (p={p_e:.3g})")
        else:
            # 全配置口径（同上，避免小样本均值相关的 p 值失效）
            rho_r, p_r = spearmanr(sub["value"], sub["effective_rank"])
            rho_e, p_e = spearmanr(sub["value"], sub["rff_error"])
            summary_rows.append({"knob": knob, "fm": "high_dim" if knob == "freq_scale" else "variational",
                                 "n": len(sub), "rho_rank_vs_knob": rho_r, "p_rank": p_r,
                                 "rho_rff_vs_knob": rho_e, "p_rff": p_e})
            print(f"  {knob}: rank vs knob rho={rho_r:+.3f} (p={p_r:.3g}) | rff vs knob rho={rho_e:+.3f} (p={p_e:.3g})")
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS, "exp04_knob_corr.csv"), index=False)

    # ============ 可视化 ============
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 行1: effective_rank vs 旋钮
    ax = axes[0, 0]
    for depth in [1, 2, 3]:
        sub = df[(df["knob"] == "freq_scale") & (df["depth"] == depth)]
        grp = sub.groupby("value")["effective_rank"].agg(["mean", "std"])
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", label=f"depth={depth}")
    ax.set_xlabel("freq_scale (high_dim)")
    ax.set_ylabel("effective_rank")
    ax.set_title("Frequency scale -> spectral rank")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for depth in [1, 2, 3]:
        sub = df[(df["knob"] == "entangle") & (df["depth"] == depth)]
        grp = sub.groupby("value")["effective_rank"].agg(["mean", "std"])
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", label=f"depth={depth}")
    ax.set_xlabel("entangle (variational)")
    ax.set_ylabel("effective_rank")
    ax.set_title("Entanglement -> spectral rank")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    sub_iqp = df[(df["knob"] == "depth") & (df["feature_map"] == "iqp")].groupby("depth")["effective_rank"].mean()
    sub_ang = df[(df["knob"] == "depth") & (df["feature_map"] == "angle")].groupby("depth")["effective_rank"].mean()
    ax.plot(sub_iqp.index, sub_iqp.values, marker="o", label="iqp")
    ax.plot(sub_ang.index, sub_ang.values, marker="s", label="angle")
    ax.set_xlabel("depth")
    ax.set_ylabel("effective_rank")
    ax.set_title("Depth -> spectral rank")
    ax.legend(fontsize=8)

    # 行2: rff_error vs 旋钮
    ax = axes[1, 0]
    for depth in [1, 2, 3]:
        sub = df[(df["knob"] == "freq_scale") & (df["depth"] == depth)]
        grp = sub.groupby("value")["rff_error"].agg(["mean", "std"])
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", label=f"depth={depth}")
    ax.set_xlabel("freq_scale (high_dim)")
    ax.set_ylabel("RFF approximation error")
    ax.set_title("Frequency scale -> classical simulability")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for depth in [1, 2, 3]:
        sub = df[(df["knob"] == "entangle") & (df["depth"] == depth)]
        grp = sub.groupby("value")["rff_error"].agg(["mean", "std"])
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", label=f"depth={depth}")
    ax.set_xlabel("entangle (variational)")
    ax.set_ylabel("RFF approximation error")
    ax.set_title("Entanglement -> classical simulability")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    sub_iqp = df[(df["knob"] == "depth") & (df["feature_map"] == "iqp")].groupby("depth")["rff_error"].mean()
    sub_ang = df[(df["knob"] == "depth") & (df["feature_map"] == "angle")].groupby("depth")["rff_error"].mean()
    ax.plot(sub_iqp.index, sub_iqp.values, marker="o", label="iqp")
    ax.plot(sub_ang.index, sub_ang.values, marker="s", label="angle")
    ax.set_xlabel("depth")
    ax.set_ylabel("RFF approximation error")
    ax.set_title("Depth -> classical simulability")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_fig = os.path.join(FIGURES, "exp04_knob_effects.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n[保存] {out_fig}")

    # ============ 关键总结 ============
    print("\n" + "=" * 70)
    print("结论（exp04 摘要）:")
    print("=" * 70)
    print("  - 查看哪些旋钮能有效调控谱结构（rank 变化大）与可模拟性（rff_error 变化大）。")
    print("  - 若 freq_scale 变大 -> rank 大/熵大，则 H2（傅里叶谱宽 -> 谱分散）得到支持。")


if __name__ == "__main__":
    main()
