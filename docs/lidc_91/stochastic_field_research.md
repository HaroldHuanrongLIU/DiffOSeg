# SSN低秩logit场：仅建议先做机制与积分精度诊断

2026-11-07。**研究记录，NOT_RUN。** 有条件值得做固定预算的小诊断，不足以直接启动完整SSN训练。它不是首次尝试空间相关采样：已有 [高斯percentile场](../lidc_field/results.md) 用共享潜变量与GED，校准最终没有采用其空间排序；[连续轮廓流](contour_flow_results.md)及[类别扩散](prior_diffusion_results.md)也未产生足够最终Dice收益。新差异应严格限定为**Bernoulli logistic-normal整掩膜似然、同边际的相关性干预及其GFM决策**。

上轮GFM样本银行的支持族oracle约+3.488 pp，而五候选oracle约+0.386 pp，实际只改变15个秩掩膜。这说明“可以生成更多掩膜”本身不是成功机制；新模型必须学到与真实边界/面积有关的联合权重，不能用更大随机支持再次证明自己。

## 一手证据与任务差异

Monteiro等NeurIPS2020的SSN令logits服从低秩加对角Gaussian，论文式(5)–(7)定义协方差和MC整图似然。其LIDC实验使用15,096张切片、约40.4%空标注，比较一份或四份未排序标注；单一确定性预测与四份标注分别求Dice后平均，不是本项目四个面积秩专用输出。原论文低秩LIDC用rank10、20个MC积分样本，并报告概率分布和Dice收益；不能把该结果直接转移为本项目90.521%→91%的预期。[原论文](https://proceedings.neurips.cc/paper_files/paper/2020/file/95f8d9901ca8878e291552f001f67692-Paper.pdf)

作者仓库的PyTorch loss先求整张图的log-prob之和，再在MC维logsumexp；LIDC实际结果来自作者链接的PHiSeg fork，同样如此。PyTorch实现明确要求类别数至少2，不能把我们的四个面积秩直接作为该softmax的四类。[作者loss](https://github.com/biomedia-mira/stochastic_segmentation_networks/blob/master/ssn/trainer/losses.py) · [作者LIDC来源说明](https://github.com/biomedia-mira/stochastic_segmentation_networks#lidc-2d) · [LIDC loss](https://github.com/MiguelMonteiro/PHiSeg-code/blob/master/phiseg/phiseg_model.py#L245-L258)

## 正确的四秩概率模型

建议首试各秩独立的空间场。令r为面积秩、i为像素，$z_r\sim N(0,I_R)$在该秩所有像素共享，$\epsilon_{ri}\sim N(0,1)$逐像素独立：

$$\eta_{ri}=\mu_{ri}+f_{ri}^{\top}z_r+\sigma_{ri}\epsilon_{ri},\qquad
Y_{ri}\mid\eta_{ri}\sim Bernoulli(\operatorname{sigmoid}\eta_{ri}).$$

每秩的协方差为 $F_rF_r^{\top}+\operatorname{diag}(\sigma_r^2)$。四个秩不是互斥类别，所有概率可同时为1；可等价理解为每秩独立二类logits `[0,η_r]`，不能跨r softmax。每张CT每秩虽只有一份标签，仍可作为条件随机场的一次完整观测训练；但四份排序标签不是同一秩的四次独立重复，旋转更不是新增读者。仅凭一例中的四轮廓不能直接估计该例的每秩协方差；学习依赖跨训练病例共享参数，也不能将均值模型的错误自动解释为纯aleatoric不确定性。

对每病例/秩，使用M个logit样本：

$$\ell_{mr}=\sum_i\left[y_{ri}\log\sigma(\eta_{mri})+(1-y_{ri})\log\sigma(-\eta_{mri})\right],\quad
L_r=-\operatorname{logsumexp}_m(\ell_{mr})+\log M.$$

随后平均病例/秩，最后除固定像素数N仅缩放目标。**不可先对像素平均再logsumexp**，那会把似然变成其$1/N$次幂；也不可每像素各做logsumexp后再求和，那只拟合边际，无法训练整掩膜相关性。该似然不能保留旧前景5倍BCE权重在乘积内部并仍称为规范化Bernoulli概率。首诊断只用此NLL。[作者LIDC显式reparameterization](https://github.com/MiguelMonteiro/PHiSeg-code/blob/master/phiseg/model_zoo/likelihoods.py#L17-L53)

若改成跨秩共享z并将4N个log-prob相加后只做一次MC-LSE，则定义的是**联合四秩panel**模型，积分更难；这是另一假设。共享z却分别做四次LSE并不能声称训练了跨秩联合似然。本次独立秩版本已足够服务逐秩可分解的平均Dice/GFM，不先增加跨秩相关参数。

## 最小机制诊断与控制

沿用拟议范围：原折0的32名训练患者，每人固定一片，四个原旋转；固定已核验患者排除OOF guide为μ，以同片CT与guide为条件训练小型方差/因子头。原E的确定性均值与解码器不在此阶段更新，也不用新预训练权重。固定rank5、256更新、batch8、16个antithetic logit draws；这些是本次工程预算，不是复现原论文LIDC配置。这里的μ是OOF guide，不能冒充正式E完整输出。

- 若头输出 `std=softplus(a)+1e-4`，则对角方差 **d=std²**。相关场用 $Fz+std\epsilon$；从同一已训练头移除相关性的控制应使用 $\sqrt{std^2+\sum_jF_{ij}^2}\epsilon_i$。不能把std直接加到方差上。
- 这项**同权重后验对角化干预**逐像素保留完全相同Gaussian均值/方差，因而保留完全相同Bernoulli边际，仅移除空间相关性。不存在新增参数量的混淆。若以后另外训练一份对角组，可保留同结构F头但只用行平方和；参数数目相同，但其最终边际和有效表达自由度不自动与相关组相同，应另列。
- F用小幅非零随机初始化。精确期望目标通过$FF^T$依赖F，F=0处对F的导数为零；有限MC在有对角噪声时可能给出非零随机梯度，不能依赖此噪声脱离零点。若所有噪声均为零并采用成对±z，首步还可能精确抵消。不要机械复用确定性残差的全零末层初始化。
- 新噪声流与病例/旋转流分离；每步刷新，正负成对的z及ε同时保持正确边际。固定μ和非零方差后，`μ>0`的确定性预测仍可原样核对；**零协方差不意味着完整Bernoulli标签抽样等于该二值掩膜**，因为σ(μ)仍通常不为0/1。

现有E的 [HierarchicalRanks](../../reproduction/hierarchical_sam.py#L116)本身已有原生CT细节和四logit输出，将来技术上可从其原生特征增加均值/方差头；本次先不改它。已有 [contour_field.population_objective](../../reproduction/contour_field.py#L79)是Gaussian percentile阈值和GED，不含上述joint NLL，所以并非代码重复，但属于已有随机场方向的受限重检验。

## Whole-mask MC是主要风险，不只是防溢出

`logsumexp`数值稳定不保证积分准确。16384个像素概率相乘，M=16甚至256也可能主要由一个“幸运”draw决定。负logmean估计对MC随机性的**期望向上偏**，但单次估计不保证是真NLL的上界；不同协方差可有不同估计偏差。只看一个新种子的256-draw分数不能将差值归因为空间相关性。

应记录每病例/秩的MC责任权重 $a_m=softmax(\ell_m)$、最大权重和 $ESS=1/\sum_m a_m^2$，并用独立随机银行查看NLL差值的波动及随预算变化的稳定性。ESS接近1表明少数draw支配，不是“训练成功”的证据。不能因训练loss下降就忽略此检查。作者原文也报告协方差发散及不稳定案例，其实现甚至会回退对角分布；本诊断若出现非有限值，应失败并记录，不能静默回退后继续宣称相关模型。[作者协方差实现及回退](https://github.com/biomedia-mira/stochastic_segmentation_networks/blob/master/ssn/models/stochastic_deepmedic.py#L29-L57)

**可验证的积分基准：** 对角化后的真实联合分布在像素间独立，故可先用一维Gauss–Hermite积分计算 $q_i=E_{\eta_i\sim N(\mu_i,v_i)}\sigma(\eta_i)$，再直接求 $-\sum_i\log Bernoulli(y_i;q_i)$。预先固定两种节点数作数值精度核对；不要用粗whole-mask MC的对角分数当真值。

相关场也可Rao–Blackwell化：条件于共享z，对独立ε逐像素积分，

$$q_i(z)=E_{\epsilon_i}\sigma(\mu_i+f_i^Tz+std_i\epsilon_i),\quad
p(y\mid x)=E_z\prod_i Bernoulli(y_i;q_i(z)).$$

这保持同一模型，MC只剩5维共享z，可显著减少独立像素噪声造成的积分方差；并不保证5维积分已收敛。它可以先作为诊断估计器。若改训练估计器须记录为实施修订，不能把精度变化与模型收益混在一起。

## 抽样与GFM接入不能偷换分布

作者PyTorch sampler以及实际LIDC评估都是Gaussian logits后argmax；二元情形相当于`η>0`。[作者sampler](https://github.com/biomedia-mira/stochastic_segmentation_networks/blob/master/evaluation/running_metrics/samplers.py#L76-L81) · [LIDC评估](https://github.com/MiguelMonteiro/PHiSeg-code/blob/master/phiseg_full_evaluation.py#L218-L233) 这是论文使用的条件模式掩膜分布，与积分Bernoulli/categorical似然的完整标签分布不同。

若本项目要求GFM对应训练模型，就必须先抽η，再逐像素抽 $Y\sim Bernoulli(\sigma\eta)$，用完整二值掩膜计算面积和GFM权重。只threshold η可作为单列诊断，但不能称为同一模型的Bayes样本；也不能以软概率代替Y、以概率和代替随机面积，因为Dice的随机分母不可这样交换期望。任何形态学清理、重新排序四个秩或按Dice筛选样本也会改变模型。

匹配对角化的单像素边际与相关场**理论上相同**。固定μ的对称Gaussian还保证边际概率与μ同侧于0.5，因此阈值边际不能改变原均值二值预测；潜在Dice收益只能来自联合面积决策，或后续另行训练mean。任何匹配两组的边际Brier差应先归为MC数值误差。有限采样GFM也可能在独立分布中看到偶然相关性，故需独立随机银行核对决策稳定性；“有限银行最优”仍不是证明真实后验最优。

## 继续或停止的结论

此256步诊断可以检查：头是否真正学习非零相关因子、方差是否有限、joint NLL差异能否在积分校验后保留、真实标签抽样是否产生新的可复现面积—位置耦合。训练Dice、样本多样性及支持族oracle均不能作为晋级证据。

若相关组仅在粗MC估计下优于同边际对角化，较准确积分不能确认，或方差持续发散，则停止当前方案。若只是增加散点噪声/随机支持，而GFM在独立银行间不稳定，也不进入完整训练。只有数值上可信的相关性效应才支持另行登记full-fit mean+cov模型与患者隔离的GFM检查；后者必须再证明实际Dice收益。32名训练患者及其旋转不能提供新的患者泛化证据。

本研究未运行诊断、训练、抽样或评分，未改运行源码或下载模型。


## 高精度诊断之后的首选患者验证（建议，尚未执行 / NOT_RUN）

已读取 [准确版诊断](/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261107_preference91_stochastic_likelihood_accurate/summary.json) 与 [独立审计](/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261107_preference91_stochastic_likelihood_accurate/audit.json)：Q128训练/评估、同边际对角Q256/512检查通过；两银行拟合NLL相对对角化降低约34.59%/34.55%，36项独立自适应积分最大误差约8.37e-6。这只建立了32名拟合患者中的数值和结构证据。以下仍是**患者验证建议，不是已执行结果或正式晋级**；科研门槛用于决定是否继续，无须解读为新增用户确认流程。

**首选固定均值的covariance校准+GFM，暂不联合训练mean/cov。** 先问同一均值下学到的相关性是否改善真实Dice决策；同时改变均值会把确定性分割修正、概率校准和相关性效应混在一起。若以后转向joint adaptation，必须配置同预算mean-only控制，不能以当前高NLL增益自动支持那条路线。

**规模和隔离：** 只做原内层折0，一个协方差头，使用该折全部inner-fit患者及图块；从原小随机初始化重新开始，不以32例诊断后的权重热启动。保持rank5、batch8、Q128条件积分、M16 antithetic latent、原优化器及四整数旋转，预先固定1024更新作为首轮预算；病例抽样沿用E现有病例/患者混合权重，不按难度选样。保留原library/check患者边界；所有教师和协方差头均排除两者及原outer-test患者，library不参与参数更新。训练结束保存固定末步，不能按check结果延长训练或挑checkpoint。

**不要直接把OOF U-Net均值换成E均值。** 首试在训练时使用原三路OOF guide；验证时分别用**这三份冻结教师**的单片预测作为均值，各自通过同一协方差头，形成三个预测分布的等权混合。复用教师原来的图像定位/视角预测函数，不能用真值中心。这样控制教师家族、权重和训练规模；E仅作为封存的目标基准。这是一个新的明确预测器，而非E的小修正。若它改善教师族却不胜E，不能宣称已改善E，也不自动把方差头移植到E；E均值的精修/融合及logit幅值分布不同，须另行用来源匹配的患者排除E式均值作校准研究。

**固定控制：** 同一训练头、同一三教师混合下逐分量作 `std_diag=sqrt(std²+sum(F²))`，保留每像素边际；称为“分量内对角化控制”，因为共享教师混合索引仍可产生组件间相关性。另报告三教师均值logit阈值和原E。混合似然应为三份整掩膜likelihood的等权和再取log，不能先平均三个NLL，也不能先平均各像素概率后把乘积当作同一混合模型。

**旋转不平均协方差基底。** 首试协方差头仅在canonical方向部署，训练继续D4增强；教师均值仍按原冻结推理规则获得。不得把四个旋转的F拉回后直接平均：F与FQ表示相同协方差，符号/正交基底不具唯一对应，平均可相互抵消。如以后加入协方差TTA，应将各视图的完整分布/二值样本逆旋转后作为预定权重混合，而非平均F或假定共享潜变量坐标已经对齐。

**数值/成本停止条件先于check计分：** 继承已通过的Q128与高阶对角精度核验，但新患者、新方差范围必须重新检查；记录std/factor尾部、责任权重ESS、Q128/256及diagQ256/512误差，不能以旧32例证据覆盖新模型。协方差持续增长、非有限值或未达到原逐像素1e-5精度门槛时停止本轮，不静默截断方差、换对角模型或挑较早checkpoint。高阶积分与采样逐病例、逐MC块执行；固定训练预算和数值检查范围，不把提高GH节点当作临床超参搜索。

**GFM只读真实二值样本。** 每教师/秩固定生成256张完整Bernoulli标签掩膜，三个分量各占1/3；再生成第二组独立随机银行，并保留每组前128张/教师的嵌套子集。模型、latent、独立像素Gaussian噪声和Bernoulli uniform的来源均封存。对角控制可复用相同独立ε和uniform以降低配对噪声，但不能把相关场总噪声标准化后当作对角ε，否则空间相关性仍在。按同一银行运行既有精确GFM和同分布最优固定边际前缀，不调温度、不加E混合权重、不筛选高Dice样本。

两个独立银行的GFM决策互相在另一银行上重算期望效用，区分真实模型下的效用与同银行取最大值的乐观偏差。library阶段只检查数值/决策稳定性并固定来源，不择优银行；主银行预先指定。建议若128→256或两独立256银行的library宏平均实际Dice差超过0.01 pp，停止当前采样预算的患者试验，不打开check分数。即使NLL已稳定，也不能假设Bernoulli/GFM决策同样稳定。

GFM可能比小银行昂贵得多。可用面积上界 $U(k)=E[2\min(k,|Y|)/(k+|Y|)]$ 对不可能胜过当前已知解的k>0作严格剪枝，k=0仍按空掩膜概率单独处理，保留并列并用小掩膜穷举核验；否则按原算法扫描完整合法范围。若成本超出预定资源预算，记录未完成，不能悄悄改成稀疏面积网格后仍称全局最优。增加随机样本几乎必然放大I/U支持，因此不再以支持族oracle单独作为继续理由。

**患者效果门槛：** 在library数值检查通过后，先封存全部check患者上相关GFM、同边际对角GFM、固定前缀、均值阈值和E的两银行输出，再读真值。首轮建议要求主银行相关GFM相对E至少+0.05 pp，且相对同边际对角GFM、同分布前缀均至少+0.02 pp；第二银行方向一致、两银行宏Dice差不超过0.01 pp。报告四秩、患者配对区间和相对教师均值的变化。NLL改善但Dice未过门槛时关闭此固定均值决策路线，不以样本多样性或oracle替代实际结果。

通过这一折也仅支持下一折/完整四折的事前登记确认；当前队列已多次用于研发。原214患者1609单片四折未舍入Dice_mean≥0.91仍是最终目标，不由上述拟合似然或单折结果代替。本补充未执行任何训练、重放、抽样或评分。
