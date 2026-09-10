"""
scripts / exp10_scaling16.py
==============================
16 qubit 完整扫描（exp06 的补充：exp06 含 8/12 qubit 全扫描 + 16 qubit 观察配置；
本实验给出 16 qubit 的完整 24 配置扫描）。

设计：
  - 2 任务（noisy_harmonic, cross_high）× 4 映射 × 3 seeds = 24 配置
  - n_qubits = 16, N = 250
  - 复用 exp06 的 run_config（同一模拟器、同一特征映射、同一基线协议）

输出：results/exp10_scaling16.csv
"""

import sys, os, importlib.util, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

# 复用 exp06 的实现（保持两实验协议完全一致）
spec = importlib.util.spec_from_file_location(
    "exp06", os.path.join(os.path.dirname(__file__), "exp06_scaling.py"))
exp06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp06)

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")


def main():
    print("=" * 70)
    print("实验 10：16 qubit 完整扫描（24 配置，N=250）")
    print("=" * 70)
    records = []
    t0 = time.time()
    for task in ["noisy_harmonic", "cross_high"]:
        for fm in ["angle", "iqp", "variational", "high_dim"]:
            for seed in [0, 1, 2]:
                r = exp06.run_config(task, 16, fm, 250, seed, verbose=False)
                records.append(r)
    print(f"耗时 {time.time()-t0:.0f}s")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS, "exp10_scaling16.csv")
    df.to_csv(out, index=False)
    print(f"[保存] {out} ({len(df)} 条)")

    from scipy.stats import spearmanr
    r, p = spearmanr(df["effective_rank"], df["rff_error"])
    print(f"16 qubit: rank~rff rho={r:+.3f} p={p:.1e}")
    r, p = spearmanr(df["best_classical"], df["quantum_advantage"])
    print(f"16 qubit: adv~classical rho={r:+.3f} p={p:.3f}")


if __name__ == "__main__":
    main()
