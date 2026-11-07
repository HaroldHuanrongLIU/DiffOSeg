# 冻结 DINO 特征的空间分辨率：四相位与 AnyUp

本页保留调研阶段的方案与核查记录。后续实施已完成两折受控试验，结果见 [完整实验报告](guided_dino_results.md)。
2026-11-07。**研究建议，NOT_RUN。** 优先试验冻结的 **AnyUp paper 权重：原始 DINOv2-S 384 维、32×32 → 64×64，再做原折 PCA**。比较相同容量的双线性上采样及同片模糊图像引导，检验细粒度图像信息是否有可部署的增量。这里的新假设是特征的空间分配，而非换更大 DINO、放大结节视野或增加另一组标注。当前正式 E 仍为 `0.9052097376196222`；没有证据保证本方法达到 `0.91`。

## 与已经完成的实验的区别

| 既有实验 | 已核对结果 | 本次不同点 |
|---|---|---|
| [局部 DINO-L 探测](dino_local_results.md) | 64 原生像素局部视野比全图 −0.001871 pp，比起始提示 −0.025102 pp | 本次保留完整 128×128 视野和 DINO-S，不裁剪放大结节。 |
| [双尺度完整解码](dino_dual_decoder_results.md) | 双尺度比全图 +0.027544 pp，但比 E −0.030521 pp | 本次不重新训练主体 U-Net，保持 E 引导，细化 Hiera 的 64 尺度特征。 |
| [DINO-S 粗层融合](semantic_fusion_results.md) | 两折比 E +0.042072 pp，未达到预定 +0.05 pp 门槛 | 保留此 32×32 粗层注入，在匹配控制中共同增加 64×64 支路，隔离上采样策略。 |

## A：同片四相位交织，几何成立但语义需验证

用同一个已注册、已按该旋转规范化的 448×448 输入作四次采样，源坐标偏移 $(\delta_x,\delta_y)\in\{-3.5,+3.5\}^2$；这里正号表示读取更右/下的源位置，必须与实现的平移方向一致。采用半像素缩放约定时，14 像素 patch 的第 $i$ 个中心映回原生坐标为

$$c_i(\delta)=(14i+6.5+\delta+0.5)/3.5-0.5=4i+1.5+\delta/3.5.$$

于是负相位给出 `0.5,4.5,...124.5`，正相位给出 `2.5,6.5,...126.5`；二维交织后恰为 64×64、中心 `0.5,2.5,...126.5` 的网格。这是采样几何推导，不是四倍真实 CT 分辨率。相位应在一次规范化后生成，固定边界延拓方式；不要每次重新估计 CT 灰度分位数。[DINO 官方 patch 卷积](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/patch_embed.py)

每次编码仍是 32×32 token，与直接把 patch stride 改为 7 并在 64×64 tokens 上做自注意力不同：每相位保留独立的全局上下文和位置编码。ViT 的位置项意味着平移前后不能假定精确等变，交织可能带来棋盘相位偏差。需在固定编码精度/逐图路径下检查相同相位重放、相位间边界跳变和输出坐标；不能用特征数量增加代替 Dice 证据。[官方位置编码与 token 路径](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) · [FeatUp 对减小 patch stride 的原始对照](https://arxiv.org/html/2403.10516v1#S6.SS2)

若以后试 A，最有解释力的控制是：同样四次编码，先按真实中心回采样到共同粗网格平均，再双线性上采样到64；与直接交织使用相同适配器、训练量和图像。另保留单次编码双线性控制。当前先不把 A 与 AnyUp 同时叠加，以免增加计算和解释歧义。

## B：AnyUp 与 FeatUp 的可用性和数据来源

| 项目 | AnyUp paper | FeatUp JBU |
|---|---|---|
| 原始方法/监督 | ImageNet 图像上，以冻结 DINOv2-S 的局部裁剪特征作目标，cosine+MSE，并有输入/增强一致性；未使用医学标注。[论文 §4.3、§5](https://arxiv.org/html/2510.12764v2#S4.SS3) | COCO-Stuff 图像，冻结编码器的随机视图特征重建；官方训练加载 `include_labels=False`，不是使用 COCO 分割标签监督这个上采样器。[配置](https://github.com/mhamilton723/FeatUp/blob/main/featup/configs/jbu_upsampler.yaml) · [训练代码](https://github.com/mhamilton723/FeatUp/blob/main/featup/train_jbu_upsampler.py#L110-L174) |
| 公开权重 | `anyup_paper.pth`，针对 DINOv2 ViT-S/14，独立于骨干；官方 `anyup` 默认入口。[hubconf](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/hubconf.py) | 官方有 DINOv2-S/14 的有/无额外 ChannelNorm 两组 checkpoint；Hub 默认同时构建并加载骨干，不能把 64 维 PCA 特征直接当作其 384 维骨干输出。[Hub 实现](https://github.com/mhamilton723/FeatUp/blob/main/hubconf.py) · [DINOv2 实现](https://github.com/mhamilton723/FeatUp/blob/main/featup/featurizers/DINOv2.py#L384-L399) |
| 形状 | `image[B,3,H,W]`、`features[B,C,h,w]` → `output[B,C,Hout,Wout]`；支持 `output_size=(64,64)` | 默认 JBUStack 连续4次×2，32输入会得到512，再缩至64不是一次×2，也不是官方默认等价配置。[JBUStack](https://github.com/mhamilton723/FeatUp/blob/main/featup/upsamplers.py#L228-L248) |
| 运行依赖/许可 | 标准非 NATTEN 推理模块为 PyTorch 路径，使用 `nn.RMSNorm`；NATTEN 可选且窗口与论文路径不同。项目声明 **CC-BY-4.0**。[依赖](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/pyproject.toml) · [许可](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/LICENSE) | 官方包需要 C++/CUDA AdaptiveConv 扩展，并锁 `timm==0.4.12`；项目 **MIT**。相对现有环境有更多安装兼容工作。[setup](https://github.com/mhamilton723/FeatUp/blob/main/setup.py) · [许可](https://github.com/mhamilton723/FeatUp/blob/main/LICENSE) |

这些都是自然图像预训练，符合本任务允许的额外来源；后续医学监督仍只能来自当前折已有 LIDC。AnyUp 的自然图像 probe 结果不证明低对比灰度 CT 的收益；原论文也说明其输出依赖粗特征的线性组合，并非恢复任意丢失信息。[AnyUp 局限](https://arxiv.org/html/2510.12764v2#S6)

**身份实查：** 只读 GitHub API 返回 AnyUp `main=351807a9c4287368732cc247f26c7c81c9139af4`；[论文权重 URL](https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth) 的 HEAD 返回 HTTP 200、`3,540,612` bytes。本轮未下载、未计算权重 SHA、未严格加载。未来需固定上述源码、下载后散列和严格加载，不自动使用 `multi_backbone` 或 NATTEN 变体。非可视化标准路径的必要本地模块为 `model.py`、`layers/{convolutions,feature_unification,positional_encoding}.py`、`layers/attention/{__init__,chunked_attention,attention_masking}.py`、`utils/{img,gaussian_derivative_initialization}.py` 及对应包初始化；不要因此把训练/可视化全套依赖安装进共享环境。[标准注意力实现](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/anyup/layers/attention/chunked_attention.py)

## 建议的一项配对试验

**输入与坐标。** 直接复用本项目 `prompt.image_input(registered_CT,k,448)` 和与之对应的原始 `384×32×32` DINO token。AnyUp 前向传入这张 `3×448×448` 图，指定 `output_size=(64,64)`；实现先在448网格编码图像，再把 query 特征平均池化至64，key池化至32，并非先把输入图缩至64。7像素 query pooling 区间映回128原生坐标的中心为 `.5+2j`，对应 Hiera 的64尺度；attention窗口也按像素中心构造。RoPE 内部使用自己的0–1坐标，保持官方定义，不擅自加坐标偏移。输出中心对齐不等于特征仅感受该点。[forward](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/anyup/model.py#L58-L85) · [窗口坐标](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/anyup/layers/attention/attention_masking.py#L21-L35)

**PCA 顺序不能交换。** 模型用原特征的通道 L2 归一化、逐通道空间卷积、softmax和通道平均产生keys；压缩、白化及截断都会改变注意力，因此通常 $U(I,P(F))\ne P(U(I,F))$。先以raw384生成384×64×64，再使用原折已经仅由拟合患者估计的同一均值/白化投影，最后截断到±6。所有组用同一个PCA，不在各自上采样结果上重新拟合。[特征统一层](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/anyup/layers/feature_unification.py) · [本地投影](../../reproduction/semantic_fusion.py#L65)

三组都保留原32尺度DINO支路，向Hiera的64尺度增加同样的 `64→48→64` 1×1残差适配器，SiLU，末层零初始化；三组从同一基础权重开始，新增参数创建不改变原随机流。先解包 Hiera 的金字塔，在64尺度加残差，保持128尺度分支和粗层相同。

- **主要组 U：** `P(AnyUp(I,F))`。
- **容量控制 B：** `P(bilinear(F,64,align_corners=False))`，同样先上采样raw384再投影/截断。
- **引导细节控制 G：** `P(AnyUp(blur(I),F))`。只模糊当前片的448输入，固定Gaussian `sigma=3.5`（对应native sigma1）、核宽23、reflect边界，不重新估计分位数；raw DINO token保持完全相同。这是细节敏感性干预，模糊也会改变对比，不应将差值全归因于边界定位。任何组不使用其他病例图像。

冻结 DINO、AnyUp、PCA、E引导。三组使用同一锁定的主损失系数、5000步、三种子、OOF提示、EMA、四旋转和`.25`最终融合；不要同时切换正在运行的 DINO+缩小主损失组合。若选该组合为共同底座，须先锁定其结果和配置，再在U/B/G三组统一使用。固定特征生成的设备、dtype、单图批量和chunk设置；按图上采样后立即投影，避免持久化整库384维高分辨率输出。

先做形状/坐标与真实模型工程烟测：严格权重加载、同图重复数值重放、四旋转数据对应、原PCA患者身份、冻结参数无梯度、零残差恢复共同初始输出；抽样独立核对先上采样后PCA。DINO-L历史曾出现批量改变造成2.31%相对L2差，不能把“冻结”当作数值重放保证。[历史诊断](dino_local_results.md)

建议预登记原内层折0、1的三组成对试验，全部掩膜封存后统一独立复算四秩Dice；报告完整融合与单独解码器。U若两折均超过冻结E、均值至少+0.05 pp，且均值超过B及G各至少+0.02 pp，再单独锁定折2、3复验。门槛属于本次设计选择，不来自AnyUp论文。未过即保留失败；内层均是重复开发队列。只有另行锁定并完成原214患者/1609单片四折评估，才判断未舍入 `Dice_mean >=0.91`。

本轮仅新增此研究记录。未安装依赖、下载权重、修改实现、训练或评分。

## Hiera 64尺度几何复核（静态补充）

核对实际公共runtime的 `utils.py`、`hieradet.py`、`image_encoder.py`、`mask_decoder.py`，四文件SHA均匹配原 `runtime.sha256`；未运行模型或按评分选择偏移。
按原128→512半像素缩放，7×7/stride4/pad3卷积的名义中心是512坐标4i；首个2×2/stride2池化单元的几何中点是8j+2，映回native为 `(8j+2+.5)/4−.5 = 2j+.125`。与AnyUp的 `2j+.5` 相差0.375原生像素，即0.1875个64网格单位。前文“对应Hiera的64尺度”仅指相同分辨率，不能当作名义采样中心严格重合。
Tiny的stages=(1,2,7,2)，64尺度取block2；全局注意力在block5/7/9，FPN仅levels2/3接收top-down。因此这条64支路本身已有局部窗口注意力和位置编码，但没有来自更深全局注意力或top-down FPN的信息；不能以“这条支路是全局/FPN混合”作为理由。[本地构造与打包](../../reproduction/hierarchical_sam.py#L88)
上述公式是卷积/池化网格的名义锚点，不是整个学习特征的精确有效感受野中心：前置窗口attention会混合多个位置，max-pool选择随通道和输入而变。后续1×1投影与pixel-unshuffle/shuffle不改变索引；官方mask decoder也直接把此64支路加到粗层转置卷积输出，没有几何重采样契约。
因此可证明“名义格点有差”，尚不能证明“当前融合有空间bug”或“必须移动AnyUp”。固定−0.375 native的采样偏移若要研究，只能另行预登记为表示融合假设，并对AnyUp、bilinear、blur三组同样应用；它不是本次失败结果的已证实修复。上述说明记录于库内分支诊断执行时；该诊断现已完成，结果见主实验报告。
