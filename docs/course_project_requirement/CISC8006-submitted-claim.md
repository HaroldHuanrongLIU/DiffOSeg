# Final Project Paper and Claim

## Paper

DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model

## Claim

**Claim — scaled-faithful reproduction of DiffOSeg Table 3.**

Under a frozen patient-level held-out fold of the public LIDC-IDRI dataset, the Stage-I categorical diffusion model using DiffOSeg’s **probabilistic consensus strategy (P)** will outperform the same model using **average annotation aggregation (A)**, achieving lower 30-sample Generalized Energy Distance (GED₃₀) and higher 30-sample threshold-aware Dice (DiceSoft₃₀).

We will preserve the paper’s Stage-I task, multi-rater labels, 128×128 preprocessing, cosine categorical diffusion schedule with 250 steps, and 30-sample evaluation; reduce only the paper’s four-fold evaluation to one pre-registered patient-level fold. Each condition will be trained with three fixed seeds under a ceiling of two RTX 5090 GPUs and 10 GPU-days. The paper reports P versus A as GED₃₀ = **0.0773 vs 0.1029** and DiceSoft₃₀ = **92.20 vs 90.43**.

The claim is **supported** if P has both lower mean GED₃₀ and higher mean DiceSoft₃₀ across the three seeds, with paired 95% bootstrap confidence intervals in the expected directions; **not supported** if either metric is significantly reversed; and **inconclusive** otherwise.

## 截图中的说明

这个版本逐项满足要求：

- 系统：DiffOSeg Stage I；
- 目标结果：论文 Table 3；
- 数据与划分：LIDC-IDRI，预注册患者级 held-out fold；
- 比较：P vs A；
- 指标与方向：GED₃₀ ↓、DiceSoft₃₀ ↑；
- tier：scaled-faithful；
- 算力上限：2×5090、10 GPU-days；
- 论文预期数值：明确列出；
- 支持/不支持/不确定规则：明确。
