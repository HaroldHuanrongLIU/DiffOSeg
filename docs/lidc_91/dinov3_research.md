# DINOv3-S/16 冻结特征替换：来源、接口与配对方案

2026-11-07。**研究记录，模型加载/训练/评分均为 NOT_RUN。** DINOv3-S/16 可作为现有 DINOv2-S 粗层融合的一项直接替换试验：输出仍是384维32×32 patch特征，训练端仍用同容量PCA与残差适配器。动机来自预训练表示的变化，不是增加医学数据、放大局部视野或再次上采样。论文的通用密集预测结果不保证本项目四秩Dice提高，更不保证达到0.91。

## 公开权重和许可

Meta的直接权重入口需要申请；其官方README同时明确列出 **timm从1.0.20起支持DINOv3**。本次采用被该入口链接的维护者发布 `timm/vit_small_patch16_dinov3.lvd1689m`，不使用第三方镜像或绕过Meta入口的授权链接。[Meta发布说明](https://github.com/facebookresearch/dinov3#pretrained-models) · [timm模型卡](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)

无凭据Hub API记录已读回：`private=false`、`gated=false`，revision为 `3bf4720a82ec2066db88137180ff1f83a675cef0`。对该revision的 [pytorch_model.bin](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/resolve/3bf4720a82ec2066db88137180ff1f83a675cef0/pytorch_model.bin) 做无凭据HEAD，返回HTTP200、86,407,250 bytes；本研究未读取权重正文。原始元数据位于 [hub_metadata.json](/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261107_preference91_dinov3/hub_metadata.json)。

| 文件 | API声明字节数 | API声明SHA-256（尚非本地下载复算） |
|---|---:|---|
| `pytorch_model.bin` | 86,407,250 | `b7e7b7d1518a4dad2d92304ff47967c3dfee806f863552a06870de1e7498e7a2` |
| `model.safetensors` | 86,362,376 | `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040` |

权重使用 **DINOv3自定义许可**，不能按DINOv2或timm软件的Apache-2.0处理。许可要求发表相关研究时承认DINO材料的使用；向第三方提供材料/派生作品时随附该许可并遵守其条款。后续若获准下载，应保留模型卡、完整许可、revision、实际文件散列。[Meta许可原文 §1](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

## 预训练监督的边界

所选模型是**web版ViT-S/16**，由固定DINOv3 ViT-7B教师蒸馏；不是S+、卫星版或FINO医学/显微领域适配模型。Meta报告LVD-1689M来自Instagram公开帖子的通用web图像池，使用DINO自蒸馏、iBOT等自监督目标，论文还说明混入少量ImageNet-1k图像。不能把未公开的web池表述为“逐图证明不含医学图像”；可以准确表述为采用公开通用web自监督预训练，没有新增任务特定医学标注监督。[Meta模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md#training-details) · [论文数据与目标](https://arxiv.org/html/2508.10104v1#S3)

Gram anchoring是教师密集表示改进的关键，但论文§5.2明确：Small学生蒸馏及其高分辨率阶段**不直接使用Gram损失**。本项目也不增加该损失；只检查其冻结表示能否补充E。医学监督继续仅用当前折现有LIDC四份面积排序掩膜，PCA仅拟合训练患者，推理仅当前单片。[论文蒸馏细节](https://arxiv.org/html/2508.10104v1#S5.SS2)

## 固定接口和数值约定

已静态核对 **timm v1.0.20**：实现位于 `timm/models/eva.py`。`vit_small_patch16_dinov3`为384维、12层、6头、4个register、动态输入尺寸、无绝对位置嵌入；`forward_features`返回经过最终norm的全部tokens。它与Meta接口不同：Meta返回字典，已有独立的 `x_norm_patchtokens`；timm返回tensor。[timm定义](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/eva.py#L2468-L2490) · [Meta接口](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py#L225-L254)

拟议调用契约如下；本轮未执行：

```python
model = timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0)
model.load_state_dict(torch.load(local_bin, map_location="cpu", weights_only=True), strict=True)
model = model.eval().requires_grad_(False)
tokens = model.forward_features(image512)
assert model.num_prefix_tokens == 5 and tokens.shape[1:] == (1029, 384)
patches = tokens[:, model.num_prefix_tokens:]
features = patches.transpose(1, 2).reshape(len(tokens), 384, 32, 32)
```

512/16=32，必须去掉**1 CLS+4 registers**，不能沿用DINOv2无register实现的 `[:,1:]`；也不能调用会池化空间的普通 `model(image)`/`forward_head`。`forward_intermediates(...,indices=[11],norm=True,intermediates_only=True)`是可用于独立核对最后一层空间输出的另一接口。[timm token路径](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/eva.py#L731-L900)

输入复用当前注册CT、每旋转规范化和ImageNet均值/标准差，仅将DINO输入尺寸由448改为512；不使用模型卡默认256中央裁剪。以原项目的双线性半像素缩放计算，DINOv3 patch中心 `(16j+7.5+.5)/4−.5=4j+1.5`，与原448/patch14的DINOv2名义中心相同。两组有相同token网格，但编码器、预训练、register及像素计算量仍不同，因此比较是完整表示替换，不是单独隔离Gram或RoPE的因果效应。[默认预处理配置](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/raw/main/config.json)

**RoPE选择提前固定为timm原生fp32 periods。** 模型卡说明Meta原权重持久化bf16 periods，timm初始化fp32 periods，二者存在数值差异；将periods做bf16再转fp32是另一实现选择，不能按内层分数选用。本试验不声称逐值复现Meta。保留模型参数/buffer为fp32，用固定autocast策略控制算子精度，避免整体 `model.bfloat16()` 意外改变periods。timm的periods是non-persistent buffer，因此仅保存checkpoint哈希不足以确定模型行为，还要记录timm版本、periods值/精度与实际缓存路径。[维护者精度说明](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m#model-notes) · [RoPE源码](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/layers/pos_embed_sincos.py#L864-L915)

当前任务环境尚未安装timm。v1.0.20的Hub和权重助手对`huggingface_hub`/`safetensors`作可选导入；因此隔离固定timm纯Python运行目录、使用本地公开bin文件与已有PyTorch严格加载，是可实施的最小路径，不需要为此调用Hub自动下载或升级共享PyTorch。其他现有torchvision/PyYAML等依赖仍须实际导入验证；这是静态可行性，不是已通过安装烟测。[可选导入](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/_hub.py#L16-L39) · [包依赖](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/pyproject.toml#L29-L35)

## 对照和验证建议

主比较：**冻结DINOv3-S/512** 对 **冻结DINOv2-S/448**；另列Hiera自身粗层的等容量控制及原E。三个新增特征源均按相同训练病例、相同采样位置独立拟合64维PCA白化/±6截断，再经过同样的 `64→48→256` 零末层残差适配器，仅加到Hiera32尺度。DINOv3不能直接复用DINOv2的PCA基底；两者通道意义不同。[现有融合路径](../../reproduction/semantic_fusion.py)

保留相同初始解码器、三种子、OOF提示、批量12、5000更新、优化器、EMA、四旋转及最终`.25`融合；主损失系数须三组一致并事前锁定，不与AnyUp细层、SAM-opt、位置偏移或新阈值同时变化。旧DINOv2/Hiera报告只在数据、来源和训练配置完全匹配时才能作为直接对照，否则需重跑匹配组。

先验证公开文件的本地散列和严格加载、1029→1024 token切分、32×32几何、两种接口一致性、同片重放/批量数值差、冻结参数及PCA患者范围、零适配初始输出。固定执行精度和逐图编码策略后再缓存，不能把不同数值路径的缓存视为同一模型。工程检查通过后才进入另行登记的患者内层成对试验，封存后独立复算四秩Dice与患者配对区间；91%仍需原214患者/1609单片四折未舍入正式结果。

本研究仅新增此文件并做无正文HEAD/静态核查；未下载权重、安装依赖、修改模型代码或运行训练。
