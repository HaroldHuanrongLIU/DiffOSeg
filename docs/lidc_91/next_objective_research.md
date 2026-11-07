# 下一项目标函数研究：逐病例、逐面积秩 Lovász-hinge

2026-11-07。**证据研究与拟议诊断，本文未运行。** 在已查`results.md`、协议和代码中没有找到Lovász/hinge试验。建议只检验一项明确变化：在E解码器的**主输出**中，将soft Dice渐变替换为逐病例、逐面积秩的Lovász-hinge；保留前景5倍BCE、`.4 raw`原监督和`.08 radius`，推理仍按原硬Dice与`.25`融合评估。不再叠加预训练骨干、轮廓后处理或融合校准。

## 证据支持什么，不支持什么

Berman等CVPR2018提出Jaccard集合损失的Lovász扩展，给出二元logit hinge及作者实现。其排序按**一份二元掩膜内部的错误margin**，不是重新分配A1–A4，也不在部署时选择面积/排序前缀，所以不重复`prefix_utility`、面积秩混合或条件票数分配。[原论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf) · [作者代码](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py)

重要反证边界：同作者参与的MICCAI2019研究在五个医学分割任务比较soft Dice、soft Jaccard、Lovász-sigmoid，通常没有发现这些metric-sensitive目标间显著差异。因此“更接近IoU”不足以支持盲目完整重训；且该研究比较的是sigmoid版本，不能当作本次hinge的直接负结果。[原始医学比较研究](https://arxiv.org/html/1911.01685v1)

本项目已有的风险监督、面积决策、联合秩分配、原生多视角、边界/距离监督、PCGrad、SAM-opt和特征上采样都未形成足够增量。[总记录](results.md) 当前`vote.mask_loss`为soft Dice加 `(1+4y)×BCE`。soft Dice对logit的导数带有`p(1−p)`，但BCE对高置信错误仍有非消失纠错梯度。**不能声称原目标无法纠正这些错误。** 可检验的假设更窄：新目标是否把有限的更新量重新分配给当前掩膜内真正影响集合重叠的margin错误，而非主要提高已有正确像素的概率。[实际损失](../../reproduction/vote_contours.py#L93)

## 精确替代项

对病例b、面积秩r单独取全部原生像素，$s_i=2y_i-1$，$m_i=1-s_i z_i$。按m降序得到排列π，令 $P=\sum_i y_i$，

$$J_j=1-\frac{P-\sum_{i\le j}y_{\pi_i}}{P+\sum_{i\le j}(1-y_{\pi_i})},\quad w_1=J_1,\quad w_j=J_j-J_{j-1},\quad L_{LH}=\sum_j w_j\max(0,m_{\pi_j}).$$

每份掩膜独立求值，再对B×4份掩膜等权平均。**不把4秩当互斥softmax类别、不跨病例排序，也不把所有batch像素混成一次IoU**；否则大图块/大前景可改变其他病例的梯度权重，偏离当前病例/秩平均目标。作者binary函数默认`per_image=True`，输入应等价reshape为`[B*4,H,W]`。参考实现是MIT许可，若移植要保留相应版权/许可。[作者binary入口及梯度](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py#L16-L28) · [许可](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE)

定义 $a_t=\min(1,(t-1)/299)$，在第1步使用原损失，第300步完全换成hinge：

$$L_t=(1-a_t)L_{Dice}(z,y)+a_tL_{LH}(z,y)+L_{BCE,5}(z,y)+0.4L_{mask}(raw,y)+0.08L_{radius}(z,y).$$

margin固定为1，替代项系数固定为1，不搜索margin、温度、ramp或融合权重。这是本项目的预登记选择，不是作者给LIDC的推荐值。适用位置是`prompt.segmentation_loss`当前的原生二元标签；配准U-Net的任意角度/尺度增强会产生插值软标签，**不能直接把同一binary公式套上去**，否则是另一个未经定义的目标变更。

## 最清楚的训练对照

保留原E作为基准。新增匹配控制用同margin=1的**类别平衡hinge**替换同一soft Dice项：有前景与背景时，各类内部平均后各占1/2；仅一类时只计算该类平均。其余BCE/raw/radius、ramp、种子、训练量、EMA完全相同。这样两种新目标都去除了sigmoid型hinge梯度衰减，主要区分Lovász的当前错误排序/Jaccard增量权重和固定类平均权重。两者不是严格梯度范数匹配，须报告尺度；不能把全部结果单独归因为排序。

Lovász-sigmoid只作**诊断控制**，不增训练臂。二元硬标签时 $|y-\sigma(z)|=\sigma(m-1)$，理论上它与hinge有相同降序排列和同一Jaccard权重；梯度额外包含`p(1−p)`，可分辨饱和解释。数值饱和可合并ties，诊断若需严格相同排序可统一按m排序。作者单通道softmax包装有类别参数约束；不要误用默认`classes='present'`丢掉空前景，按明确的单前景公式实现/核验即可。[作者二元示例](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/demo_binary.ipynb)

## 必须明确的边界行为

- **没有前景并非没有有效像素。** 当P=0且像素数非零，公式退化为 `max_i relu(1+z_i)`，惩罚最坏假阳性；不能跳过。全前景时退化为平均`relu(1−z)`。仅当全部像素被ignore、有效数为0时，返回依附计算图的零。空掩膜的正式Dice仍用原评分约定，不能更换指标。[作者空输入/排序实现](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py#L86-L117)
- **ties。** 作者用`torch.sort`，未显式固定稳定性。建议FP32原始margin排序、固定row-major并列次序、`stable=True`，先排序再ReLU；不加随机抖动或标签优先规则。并列的loss值不依赖并列内次序，但可选子梯度会变化；这是可重放约定，不是宣称唯一梯度。计数、cumsum、除法、权重差分均用FP32，避免BF16计数/差分失真。margin恰为0按PyTorch ReLU的零导数约定。
- **IoU不是验收指标。** 单掩膜 $D=2J/(1+J)$ 单调，但平均Dice与平均IoU并非相同目标；本方案是Jaccard替代项，不能叫“精确优化当前平均硬Dice”。所有晋级仍按独立复算原Dice，IoU只作附加读数。

## 小诊断先于12模型训练

固定原折0、首种子已训练E的32名拟合患者，每人选固定一例；不读取库/检查真值。冻结模型，比较原目标、Lovász替代、balanced-hinge替代及仅诊断的Lovász-sigmoid。按相同实际logits求梯度，按面积秩报告：方向余弦、L1/L2范数、错误像素比例、错误/正确且margin内外的梯度质量、空前景数及ties比例。

应区分 **主输出梯度** $g=\partial L/\partial z$ 与全部可训练参数梯度；`.4 raw`分支不经过z，不能将前者称为所有参数的完整训练梯度。原/替代目标对z的梯度均含保留的BCE及radius。除绝对错误像素质量外，建议以纠错方向的质量

$$M=\frac{\sum_{i\in\mathcal E}\max(0,-s_i g_i)}{\sum_i|g_i|},\qquad\mathcal E=\{i:(z_i>0)\ne y_i\}$$

作为门槛读数，并单列错误方向质量；否则radius项产生的反向梯度也会被误当“更多纠错”。所有指标以逐掩膜/病例先计算再平均的规则固定，不能把像素池化统计混作当前目标。

拟议工程门槛：两种hinge均有限非零；Lovász与balanced的主输出梯度余弦均值<0.99；Lovász的M均值至少为原目标的1.25倍，且反向错误质量不增加。该门槛只证明可观察的监督信号变化，不证明可学性或泛化；若原M已>0.8，则1.25倍门槛几乎不可满足，应据实停止/报告而不是看结果后调门槛。原M为零、无错误或梯度总量为零时分别报告，不能除零后当作无限改善。若两种hinge仅梯度尺度不同而方向近同，或变化主要来自少量空前景，不能以此支持完整计划。

先用独立小掩膜枚举验证Jaccard增量、空/全前景、单像素、完美margin、ties的loss不变性和非并列点数值梯度；真实诊断必须运行实际输出路径。通过后才另行预登记原折0/1、三种子、Lovász和balanced两组共12模型、各5000步；原主输出BCE/raw/radius和固定`.25`推理均保持。两折全部封存后按原掩膜独立复算。建议Lovász比E两折均正、平均至少+0.05 pp，且比balanced平均至少+0.02 pp才登记其余折确认。未过即停止，不新增loss网格或改排序范围。

结论：Lovász-hinge在本仓库是尚未检查过的目标机制，但目前只有一般文献依据；必须先通过上述本地梯度诊断，才比泛用换损失搜索多一层可证伪证据。本研究未运行任何诊断、训练或新评分。
