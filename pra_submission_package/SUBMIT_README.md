# PRA Submission Package — 使用说明

目标期刊：**Physical Review A**（APS，美国物理学会），期刊类别：aps·pra。

## 用法（Overleaf，推荐）

1. 打开 overleaf.com → 注册/登录（可用邮箱/Google 账号）
2. New Project → **Journal Article** → 选择 **"APS Journal" / "RevTeX 4.2" 模板**（官方模板，自带 revtex4-2）
3. 上传本文件夹全部文件：
   - `pra_main.tex`  → 用它**替换**项目里的 main.tex（把内容粘贴进去或重命名上传）
   - `refs.bib`      → 上传到项目根
   - `fig1_difficulty.pdf` … `fig5_alignment.pdf` → 上传（共 5 张图）
4. 编译：右上角 Recompile（默认 pdflatex，Overleaf 字体完整，无需任何配置）
   - 若提示缺文件：确认文件名与 pra_main.tex 中 `\includegraphics{...}` 一致
5. 下载 PDF → 这就是投稿稿

## 投稿到 APS（Physical Review A）

地址：https://submit.aps.org （或从 journals.aps.org/pra 的 "Submit a manuscript" 入口）
- 注册 ORCID（普通个人账号即可）
- 稿件类型：Regular Article
- 上传编译好的 PDF（标题页含作者/单位/邮箱）
- 必填信息：suggested referees（可从文献引用作者里推荐 2-3 人）、题目、摘要
- **无需 arXiv 预印本**（APS 不要求）

## 文件清单

| 文件 | 用途 |
|------|------|
| pra_main.tex | PRA 版论文源码（RevTeX 4.2，aps/pra/preprint 样式） |
| refs.bib | 参考文献（apsrev4-2 样式） |
| fig1-5.pdf | 全部为矢量图，直接嵌入 |

## 与 Quantum 版区别

- 文档类：`\documentclass[aps,pra,preprint]{revtex4-2}`（APS 标准）
- 参考文献样式：apsrev4-2（数字+DOI 链接）
- 内容与 Quantum 版 v4 完全一致（12 页正文、5 图、4 表、作者信息+AI 披露+仓库链接）
