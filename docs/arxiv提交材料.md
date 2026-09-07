# arXiv 提交材料（2026-09-07）

> 用途：获得背书/机构邮箱豁免后，直接按本文件内容填写 arXiv 提交表单。

## 标题（Title）

What do spectral criteria of quantum feature-map kernels actually predict? Classical
simulability, generalization advantage, and the role of task difficulty

## 作者（Authors）

Xianzhe Liu (刘贤喆)
School of Physics and Electronics, Hunan University, Changsha 410082, China
liuxianzhe@hnu.edu.cn

## 摘要（Abstract，200 词）

直接用论文摘要（quantum_main.tex 中 \begin{abstract}...\end{abstract} 全文复制即可）。
arXiv 摘要字段允许粘贴，无需改写。

## 主题分类（Primary / Cross-list）

- Primary: **quant-ph**（量子物理）
- Cross-list: **cs.LG**（机器学习）；可选 **stat.ML**

## 备注（Comments，推荐附上）

```
12 pages, 5 figures, 4 tables. Code and data publicly available at
https://github.com/python123flask/quantum-kernel-spectral
```

## 作者提交时的附加选项

- 建议选择 "no journal reference / no DOI"（尚未发表）
- License：选择 CC BY 4.0？（arxiv 默认的许可，提交时可选）

## 提交后

1. 用论文 LaTeX 源码+quantumarticle.cls 打包上传（arXiv 支持 LaTeX 直接编译；
   模板要求 \pdfoutput=1——注意：我们本地用 xelatex 编译的，arXiv 用 pdflatex。
   投 arXiv 前把 quantum_main.tex 第 1 行加回 `\pdfoutput=1` 并删除
   `nopdfoutputerror` 选项即可在 arXiv 上正常编译，因为 arXiv 的 pdflatex
   环境字体完整。）
2. 代码仓库保持公开（https://github.com/python123flask/quantum-kernel-spectral）
3. 提交后把 arXiv ID 回填到论文 Data availability 声明（可选，建议）

## 背书流程（若需要）

- 请背书人登录 arxiv.org → https://arxiv.org/auth/endorse → 输入你的
  邮箱 + arXiv 用户名 → 确认
- 机构邮箱豁免：用 hnu.edu.cn 注册后，提交时若系统未要求背书即已豁免
