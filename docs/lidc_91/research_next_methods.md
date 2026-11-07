# 下一项研究建议：对现有解码器做 Sharpness-Aware Minimization

本页保留最初调研建议。后续 SAM 四折训练和融合校准已经完成，结果见 [完整实验报告](sam_optimizer_results.md)。
2026-11-07。**状态：研究建议，NOT_RUN；没有实现、训练或新评分。** 建议只预登记一个新假设：保持 E 的模型、监督和推理不变，改用参数邻域鲁棒优化。这里简称 **SAM-opt**，避免与 Segment Anything 模型混淆。当前正式 E 为 `0.9052097376196222`，距原单片四折 `0.91` 尚差 `0.0047902623803778`；论文不支持预测本方案能补齐此差距。[本地正式结果](outer_stable_refinement_results.md)

## 新意、依据与边界

本轮检索了 `docs/lidc_91/*protocol*`、结果汇总和 `reproduction/`：已有 PCGrad、训练轨迹 EMA、同种子不同监督参数平均、独立主体集成、各种损失及特征融合，但未发现 SAM-opt、ASAM 或参数邻域最坏损失训练的登记/实现。这是“在已检查材料中未发现”，不是对未归档实验的保证。[PCGrad](gradient_protocol.md) · [轨迹平均](trajectory_sam_results.md) · [参数平均](position_distribution_results.md) · [主体集成](core_ensemble_results.md)

Foret 等的 ICLR 2021 原论文提出最小化参数小邻域内的最大训练损失，并在自然图像分类及迁移任务上报告泛化改善；官方代码提供两次梯度评估的实现。该证据没有覆盖本项目的面积秩分割、现有复杂 E 组合或 91% 门槛。[原论文](https://openreview.net/pdf?id=6Tm1mposlrM) · [作者代码](https://github.com/google-research/sam)

我们的假设是：在约百名拟合患者上，解码器可能过度利用当前提示与少数易学特征；约束参数扰动后的同一病例损失，可能比增加后处理更有利于新患者。这个解释尚未由 LIDC 证据建立。ICLR 2024 的原始研究提出 SAM 也可能通过平衡不同特征的学习起作用，因此不能把“更平坦”直接当成“必然泛化更好”。[特征学习研究](https://proceedings.iclr.cc/paper_files/paper/2024/hash/615ce9f03a2b0174d21ee1ffa272fadd-Abstract-Conference.html)

## 两个可实施位置

| 位置 | 可复用入口与约束 | 判断 |
|---|---|---|
| **E 的三个 Hiera-Tiny 解码器** | `prompt_preference.train` 已集中拥有采样、损失、AdamW、EMA；`hierarchical_sam.HierarchicalRanks` 接收固定单片特征和真实患者排除提示。需要在原训练循环的更新边界加入两次前后向；冻结公共图像特征、固定引导和最终 `.75G+.25mean(z)` 均可原样使用。 | **优先。** 对完整推理链影响最小，能隔离优化算法。代价是每成员训练约两倍梯度计算；推理成本不变。 |
| 配准主体 U-Net | `aligned_preference.learn` 使用 GroupNorm，没有该模型内部的提示随机性；相同 v1 初始化、6000 更新及原监督可应用 SAM-opt。 | 暂缓。更强主体会改变连续引导，须复用 `core_ensemble.lock` 的差值注入与重新解码路线，另做库内强度选择；这会把优化收益与提示分布变化混在一起。 |

代码位置：[解码器训练](../../reproduction/prompt_preference.py#L204)、[解码器内部随机性](../../reproduction/hierarchical_sam.py#L136)、[主体训练](../../reproduction/aligned_preference.py#L62)、[主体变更后的重解码](../../reproduction/core_ensemble.py#L126)。未来实现应使用新的冻结实验源码；不要改动正在运行的 DINO-Small + 主损失缩小实验的快照或进程。

## 精确主方法

以 E 原始总监督为 $L_B(\theta;\xi)$；$B$ 是已抽取的病例，$\xi$ 包含本次全部随机增强/提示。保持其主掩膜、`.4 raw`、`.08 radius` 系数不变。对全部 `requires_grad=True` 且具有梯度的参数共同计算一个全局范数：

$$
g=\nabla_\theta L_B(\theta;\xi),\qquad
\epsilon=\rho\,g/\max(\|g\|_2,10^{-12}),\qquad
h=\nabla_{\theta+\epsilon}L_B(\theta+\epsilon;\xi).
$$

第二次反传不对 $\epsilon$ 求导。先完成该反传，再把参数从保存的原值精确恢复到 $\theta$，对 $h$ 做原来的范数 4 裁剪，最后只执行一次 AdamW 更新和一次 EMA 更新。若第一次梯度全零，扰动为零。原 AdamW 解耦权重衰减只执行一次，不加进扰动目标。[官方 `get_sam_gradient` 与裁剪/更新顺序](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py#L484-L522)

固定 `rho=0.05`。这是我们参照作者 CIFAR 训练命令作出的单值预登记选择，**不是论文统一默认，更不是已验证的 LIDC 最优值**；官方代码的 `sam_rho` 默认实际为 0。保留本仓库 AdamW：初始学习率 `0.00016`、权重衰减 `0.0002`、80 步预热及现有余弦日程。采用 AdamW 是为了隔离新增邻域步骤；作者实现的 SGD/RMSProp 与本次适配要明确区分。[官方示例与参数](https://github.com/google-research/sam#how-to-train-from-scratch) · [官方默认参数](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py#L38-L84)

**随机流是关键实现条件。** 每步病例、旋转、引导幅度与偏置仅抽一次。进入第一次模型 forward 前保存 CPU 和实际 CUDA 设备的 RNG 状态，结束后保存其后状态；第二次 forward 前恢复前状态，第二次 backward 后恢复第一次的后状态。这样两次 forward 使用相同框抖动和 dense-prompt dropout，且下一训练步的随机流仍与普通 E 相同。代码中的随机框在 `boxes_from_logits`，23% dense 提示丢弃在 `HierarchicalRanks.forward`。只复用外层 `g/x/y` 而不处理这两处，会把额外随机提示混入优化比较。[随机框](../../reproduction/prompt_preference.py#L120) · [提示丢弃](../../reproduction/hierarchical_sam.py#L140)

EMA 仍为每个真实优化更新后 `lerp(..., max(.005,1/step))`；禁止在扰动参数上更新 EMA、保存检查点或执行 AdamW。冻结参数、缓存、公共权重及来源校验不参与扰动。建议把更新选择放在 `prompt.train` 当前 `zero_grad/backward/clip/step` 边界，并保留其默认路径；不要仅用现有 `paired_loss` 回调在构图后恢复参数，它可能触发 autograd 版本错误或隐藏错误梯度。

## 配对验证与停止条件

- **同更新控制 A：** 普通 AdamW，原 5000 更新。**主方法 S：** SAM-opt，5000 更新，每步同一批次两次梯度。两者使用同一初始权重、病例/提示随机流、批量 12、三种子 `50171+fold+(0,17000,34000)`、学习率和 EMA；主比较为 S−A。
- **梯度预算控制 C：** 普通 AdamW 10000 更新，10000 次梯度评估，与 S 相同。按总更新比例延长原学习率日程，固定末步 EMA，保持每次更新的 EMA 规则；增加的数据抽样、优化更新次数和有效 EMA 窗口占比应明确报告。它是资源控制，不能同时声称与 S 具有相同更新数或相同采样序列。记录实际 GPU 时长，二者未必等墙钟预算。
- 先用真实模型做五步工程烟测：`rho=0` 的两次梯度路径与普通路径在参数、optimizer、EMA 和下一步抽样上相符；两次随机框/dropout 完全相同；扰动范数符合 `rho`；冻结参数不变；用独立小型二次函数检验扰动方向、恢复顺序及一次 AdamW 更新。不得用源码字符串匹配代替这些行为检查。
- 烟测通过后，建议四个原内层折的 A/S/C 全部预先锁定，三种子均重训；保留冻结 E 作为附加参照。使用固定 `.25` 融合、原四旋转推理及现有保存/平均 logits 规则，不同时搜索阈值、融合比例或 `rho`。全部检查掩膜封存后统一评分，按原面积稳定排序的四掩膜独立复算，报告逐秩、逐折、完整方法和单独解码器结果、患者配对区间。
- 建议晋级门槛：S 相对冻结 E 及 A 的四折平均均至少 `+0.05` 个百分点，至少三折为正且任一折不低于 `−0.05`；S 相对 C 平均至少 `+0.02` 个百分点。未满足即记录失败，不在同一检查结果上调整 `rho` 后重新宣称验证。门槛是本次设计选择，不来自论文。
- 只有通过内层预登记门槛后，才另行锁定原 214 患者、1609 单片的完整外层训练/推理。医学监督仍限现有 LIDC，输入仍限当前切片，不增加患者身份、邻片或医学外部数据。内层队列和原外层均已经反复用于开发/复评，不能称为新的独立外部验证；最终成功仍只看原未舍入四折 `Dice_mean >= 0.91`。

本记录只完成文献与静态代码核对；所有拟议数值测试、GPU 烟测、训练和验收均未运行。
