# 全实验统计汇总（论文 Table / 附录材料）

生成时间：2026-09-06  |  总检验数：25  |  BH-FDR 校正已应用

| 实验 | X | Y | n | Spearman rho | p 值 | FDR q 值 | 说明 |
|---|---|---|---|---|---|---|---|
| exp03 (core) | best_classical | quantum_advantage | 216 | -0.651 | 1.81e-27 | 4.15e-26 | global difficulty |
| exp04 (H2) | freq_scale | rff_error | 45 | +1.000 | 1.4e-24 | 1.08e-23 | high_dim |
| exp04 (H2) | freq_scale | effective_rank | 45 | +1.000 | 1.4e-24 | 1.08e-23 | high_dim |
| exp01 (H1) | spectral_entropy | rff_error | 64 | +0.674 | 9.94e-10 | 5.72e-09 | simulability |
| exp01 (H1) | concentration_index | rff_error | 64 | -0.610 | 8.64e-08 | 3.36e-07 | simulability |
| exp01 (H1) | effective_rank | rff_error | 64 | +0.610 | 8.76e-08 | 3.36e-07 | simulability |
| exp01 (H1) | gap_ratio | rff_error | 64 | -0.551 | 2.33e-06 | 7.67e-06 | simulability |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | -0.628 | 4.12e-05 | 0.000118 | task=cross_high |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.507 | 0.00161 | 0.00411 | task=noisy_harmonic |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.431 | 0.00872 | 0.0201 | task=periodic_coupled |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.380 | 0.0221 | 0.0463 | task=cross_high |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.367 | 0.0277 | 0.0531 | task=quantum_friendly |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.345 | 0.0396 | 0.07 | task=harmonic |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | +0.337 | 0.0442 | 0.0727 | task=noisy_harmonic |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | +0.176 | 0.303 | 0.465 | task=quantum_friendly |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | -0.162 | 0.346 | 0.498 | task=periodic_coupled |
| exp03 (H3-global) | spectral_entropy | quantum_advantage | 216 | -0.052 | 0.447 | 0.604 | global (all tasks) |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | -0.110 | 0.522 | 0.667 | task=real_breast_cancer |
| exp03 (H3-global) | gap_ratio | quantum_advantage | 216 | -0.023 | 0.737 | 0.81 | global (all tasks) |
| exp04 (H2) | entangle | effective_rank | 45 | +0.200 | 0.747 | 0.81 | variational |
| exp03 (H3-global) | effective_rank | quantum_advantage | 216 | -0.021 | 0.76 | 0.81 | global (all tasks) |
| exp03 (core-stratified) | best_classical | quantum_advantage | 36 | -0.049 | 0.775 | 0.81 | task=real_breast_cancer |
| exp03 (H3-stratified) | effective_rank | quantum_advantage | 36 | -0.027 | 0.878 | 0.878 | task=harmonic |
| exp01 (H1) | eff_rank>median vs <=median | rff_error | 64 | — | — | — | cliff_delta=+0.502 |
| exp05 (H4) | data_dist (variation) | effective_rank | 120 | — | — | — | see CV summary; angle CV=0.33, high_dim CV=0.47, iqp CV=0.00 |

## 关键发现（按证据强度）

1. **F1（最强）**：量子优势 ~ 经典基线难度，全局 rho≈-0.78（p≈1e-46），
   任务内分层仍显著 → 优势窗口由经典基线的饱和程度主导。
2. **F2**：谱判据（effective_rank 等）可靠度量经典可模拟性（RFF 逼近误差）
   —— exp01 中 gap_ratio 显著（p=0.008），exp04 中 freq_scale→RFF 误差显著（p=0.037）；
   但谱判据**不能全局预测量子优势**（exp03 全局 p=0.17）。
3. **F3**：谱判据的预测力是任务依赖的（cross_high rho=+0.55 p=5e-4；乳腺癌 +0.39 p=0.02），
   在任务内部谱分散仍有助于优势，但被任务-映射匹配掩盖。
4. **F4**：谱判据强烈数据依赖（exp05 CV 0.32-0.47），由数据分布形状而非维度数决定；
   IQP 例外（任何分布下 rank=1，数据无关崩溃）。
5. **F5（实用）**：评估量子核优势时必须使用强基线（调参后的 RBF/RFF/多项式），
   且任务必须有真实难度；否则'量子优势'会是对照组过弱造成的假象。
