# 课题：量子特征映射的经典可模拟性谱判据

> Towards a Spectral Criterion for Classical Simulability of Quantum Feature Maps


---

## 课题一句话

给定量子特征映射，能否在**训练前**用其核矩阵的**谱结构**高效预测量子核的**经典可模拟性**——即它能否被 Random Fourier Features / Nyström 低秩经典核匹配，以及它的"量子泛化优势"何时存在、何时失效。


## 核心假设（最终验证状态，2026-09-06）

| 假设 | 内容 | 最终状态 |
|------|------|---------|
| **H1** 谱-可模拟性 | 核谱集中度 → 经典可模拟性 | ✅ 强成立（5 判据全显著，top5_energy rho=-0.80, p=1.2e-15） |
| **H2** 傅里叶-集中 | 傅里叶谱宽 → 谱分散 → 难可模拟 | ✅ 强成立（freq_scale→RFF rho=+1.000, p=1.4e-24） |
| **H3** 优势-谱 | 量子泛化优势与谱分散度正相关 | ❌ 彻底证伪（全局 p=0.76；任务内方向不一致） |
| **H4** 数据依赖 | 谱判据数据依赖（分布形状主导） | ✅ 成立（CV=0.06~0.38） |
| **H5** 优势-难度 | 优势窗口由经典基线饱和程度主导 | ✅ 强成立（rho=-0.651, FDR q=4.2e-26，全项目最强） |

## 项目结构

```
quantum-ml-classicality/
├── docs/
│   └── 01-课题设计书.md          # 课题设计书（北极星文档，含最终假设状态）
├── paper/
│   └── main.pdf                  # 论文初稿（8页英文，可直接投递前修改）
│   └── main.tex                  # LaTeX 源码
├── src/
│   ├── simulator.py              # 自研量子状态向量模拟器（纯numpy）
│   ├── feature_maps.py           # 4种量子特征映射
│   ├── spectral_criteria.py      # 谱判据（核心学术贡献）
│   ├── classical_baselines.py    # 经典对照（RFF/Nyström/RBF/线性/多项式）
│   ├── tasks.py                  # 可控任务生成器（6类任务）
│   └── stats_utils.py            # 统计工具（BH-FDR/Cliff's delta）
├── scripts/
│   ├── exp01_feasibility.py      # 实验1：谱判据 vs 可模拟性（H1）
│   ├── exp02_generalization_advantage.py  # 实验2：单任务谱判据 vs 优势
│   ├── exp03_spectral_threshold.py       # 实验3：跨任务谱判据 vs 优势（H3）
│   ├── exp03_supp_difficulty.py  # 实验3补充：优势 vs 经典难度（H5核心）
│   ├── exp04_design_knobs.py     # 实验4：设计旋钮→谱结构（H2）
│   ├── exp05_data_dependence.py  # 实验5：数据依赖（H4）
│   └── summary_all_experiments.py # 全实验统计汇总（BH-FDR）
├── tests/
│   └── test_simulator.py         # 模拟器单元测试（含幺正性回归测试）
├── results/                      # CSV结果 + 结果备忘（03为最终版）
├── figures/                      # 实验图
├── data/                         # 数据集（预留）
└── papers/                       # 参考文献（预留）
```

## 如何运行

```bash
# 1. 验证模拟器正确性
python3 tests/test_simulator.py

# 2. 实验1：谱判据 vs 经典可模拟性（H1）
python3 scripts/exp01_feasibility.py

# 3. 实验2：谱判据 vs 量子泛化优势（H3）
python3 scripts/exp02_generalization_advantage.py
```

**依赖**：numpy, scipy, matplotlib, scikit-learn, pandas。纯 CPU。

## 已获得的关键结果（里程碑1）

1. **自研模拟器全部单元测试通过**，正确实现单/双比特门、保真度核。
2. **不同特征映射谱结构差异巨大**：effective_rank 从 1.0（IQP，完全集中）到 9.0（angle 大样本）。
3. **H1 初步成立**：有效秩低组 RFF 逼近误差 0.61 vs 高组 4.43（差 7 倍）；gap_ratio 显著预测（rho=-0.331, p=0.008）。
4. **H3 初步成立**：spectral_entropy (rho=+0.415, p=3e-4)、effective_rank (rho=+0.354, p=2e-3) 显著预测量子优势方向。
5. **重要发现**：在简单合成任务 + 4 qubit 上，经典核全面超过量子核（量子优势≤0 占比100%）。说明量子优势需要"谱分散核 + 足够难/真实任务"，这指明了下一步研究关键。

## 下一步（第 2 月目标）

1. 构造**真实结构任务**（高维非线性、群对称、UCI 数据集），让量子优势有浮现空间。
2. 扫描**谱判据阈值窗口**，定位量子优势"相变"区（课题最具原创性的贡献点）。
3. 引入**数据依赖判据**，研究同一映射在不同数据上的谱变化。
4. 补齐**统计严谨性**（BH-FDR 多重比较校正、效应量、功率预计算）。

## 第二轮增强（2026-09-07）：规模/真实数据/噪声/量规

针对评审短板的四条升级，全部完成，详见 results/实验结果备忘04-第二轮增强.md：

| 增强 | 结果 |
|------|------|
| exp06 扩规模（8/12/16 qubit） | H1 随规模**增强**（ρ=+0.80）；角编码有效秩不随规模集中（≈9.5 恒定） |
| exp07 真实数据（iris/wine/breast/digits） | **0/48 正优势**（经典 0.97-1.00 饱和）；H1 达 ρ=+0.97 |
| exp08 噪声（退极化 0-30%） | 全部结论稳健；噪声提高 RFF 误差（更难逼近）但不翻转结构 |
| exp09/09b 量规尝试 | 谱展宽比无效（p=0.13）；**核-目标对齐全局显著（ρ=-0.42）但控制难度后完全归零（p=0.99）**——文献的核心预测量被难度吸收 |
| exp10 16 qubit 完整扫描 | H1 保持（ρ=+0.776）；**有效秩与规模完全无关**（angle 恒 9.62 等） |
| exp11 振幅阻尼 | 强烈压缩有效秩（11.8→2.2）并**破坏 H1**（ρ=+0.04 vs 退极化 +0.35）——噪声类型决定判据有效性 |

## 投稿版论文

```
paper/
├── quantum_main.pdf       # 投稿版初稿（9页，Quantum样式，已编译验收）
├── quantum_main.tex       # LaTeX 源码（quantumarticle 文档类）
├── refs.bib               # 参考文献（quantum.bst 风格，DOI/arXiv 链接）
├── quantumarticle.cls     # 期刊模板类文件（投稿需随 arXiv 上传）
├── quantum.bst            # 期刊 BibTeX 样式
├── main.pdf/main.tex      # 旧版 article 样式（备份）
└── main_article_backup.tex
```

编译命令（本地环境）：
```bash
cd paper
export TEXINPUTS=~/texmf/tex//:   # 本地安装 revtex 组件用
xelatex quantum_main.tex && bibtex quantum_main && xelatex quantum_main.tex && xelatex quantum_main.tex
```

## 图片（论文全部使用 PDF 矢量图）

```
figures/pdf/
├── fig1_difficulty.pdf    # 量子优势 vs 经典基线难度（核心图）
├── fig2_simulability.pdf  # 谱判据 vs RFF 可模拟性
└── fig3_freqscale.pdf     # 频率尺度旋钮效应
```

重新生成：`python3 scripts/make_pdf_figures.py`（从 results CSV 重绘，矢量输出）。
