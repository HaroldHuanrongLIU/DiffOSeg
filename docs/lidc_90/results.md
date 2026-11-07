# LIDC 单切片偏好平均 Dice 达到 90%

2026-11-06。预先锁定的面积修正候选，在原始 214 位患者、1,609 张 128×128 CT 图块上的四折等权偏好平均 Dice 为 **90.3481%**，v9 为 **89.1872%**，增加 **1.1608 个百分点**。目标按未舍入点估计判定，原始值为 `0.9034805187894748`。四折各自的点估计均超过 90%。按折分层、以患者为单位的 5,000 次配对 bootstrap 得到绝对值 95% 区间 **89.7657%–90.8936%**，改变量区间 **+0.3424 至 +2.5343 个百分点**。绝对区间下限低于 90%，尚不能以该置信水平断言总体均值高于 90%。

| 折 | 患者 | 图块 | v9 偏好 Dice (%) | 本轮偏好 Dice (%) | 差值 (pp) |
|---|---:|---:|---:|---:|---:|
| 0 | 54 | 450 | 89.6419 | 90.1262 | +0.4843 |
| 1 | 54 | 375 | 87.1197 | 90.5557 | +3.4361 |
| 2 | 53 | 412 | 90.2571 | 90.3702 | +0.1131 |
| 3 | 53 | 372 | 89.7303 | 90.3401 | +0.6098 |

## 方法与单切片约束

1. **先校准坐标再做数据增强。** 只在本折训练患者上，比较四人中间投票轮廓内外窄带的 CT 强度差，在水平、垂直各 −2 至 +2 像素的网格内估计共同位移。先按该位移平移 CT，再做空间增强。固定原标注和原评分坐标，四个全折拟合均得到向左、向上各一像素。这是此数据版本的经验校准，无法直接推断原始 DICOM 或其他医院也有同样的偏移。
2. **先在整张图块中找结节。** 定位器经过宽范围水平平移训练，从当前切片估计结节位置。预测接近中心时保留固定图块中心，明显偏心时把视野移到图像预测的位置。定位器输出仅从 CT 获得。配准后的全视野分支从匹配折的 v1 权重初始化，学习四种按面积排列的模拟偏好分割，并加入原生面积与质心监督。
3. **在原像素网格上精修面积。** 放大视野对同一 CT 的两种训练变换逆映射后比较真实掩膜。另用本折训练患者的扰动轮廓和 CT 内外强度剖面拟合多输出回归树，预测四种等效半径。最终按图像模型的置信度次序少量增加或去除边界像素。每个折只通过内层库患者选择融合和半径收缩强度。推理只输入目标切片和从该切片产生的定位、轮廓及已冻结的单切片基线预测。没有使用真实面积、邻片、患者编号、真实标注或 alpha 作为预测特征。
4. **明确区分开发支线。** 初期 ConvNeXt 局部模型使用了 ImageNet 初始化及仅训练时的 alpha 辅助监督，其局部预测没有被选为最终偏好模型。在该阶段训练的独立全图定位器从头训练，只用训练掩膜提供定位和共识监督。内层“已知真实面积”诊断约为 91.5%–91.8%，只用于估计误差来源。它没有进入候选选择或正式评分。

## 主要贡献与控制

| 外层测试配置 | 四折平均偏好 Dice (%) |
|---|---:|
| v9 | 89.1872 |
| 配准基础分支 | 90.3746 |
| 加入放大视野 | 90.3426 |
| 加入面积修正，测试前选定的主方法 | 90.3481 |
| 主方法改用固定中心，权重相同 | 89.9341 |

锁定前内层四折等权的基础、放大、面积修正分别为 90.3344%、90.3473%、90.4192%。按注册规则选择面积修正。外层基础分支略高于最终候选，按约定保留先选定的 90.3481%。额外的放大与面积修正没有显示出超过基础分支的外层净收益。最明显的进步出现在折 1，相对 v9 增加 3.4361 个百分点。固定中心时该折为 88.1110%，启用同一张 CT 上的定位后为 90.5557%；定位在折 0 和折 2 有小幅退步。折 0、1 中，同一初始化与预算下比较配准和零位移的内层结果分别为 90.1125% 对 89.5585%、90.3398% 对 89.9615%。这些对照说明了定位和训练配准的作用，但不足以把总体提升精确分摊到每个模块。

A1–A4 的四折平均 Dice 分别为 89.5525%、90.6030%、90.9478%、90.2888%。目标覆盖四偏好平均，A1 单独仍低于 90%。v9 共识分布可继续与新偏好输出配套使用，本次只评价偏好目标。训练、选参和测试延续原患者级划分，四个训练种子由折号确定。此前多轮研究已使用相同的 LIDC 队列，本轮是同队列复评。

## 核验与复现

- 所有模型选择记录及 `selection.json` 早于任何外层预测；四折全部封存后才对照外层标注评分。独立程序遍历全部 1,609 例，分别为五种方法、四个面积等级重新计算 32,180 个 Dice 值，最大差为 0。加入原生区域顺序、患者互斥、冻结哈希和 v9 回算检查。另在每折取一个实际图块，只提供图像，替换同批另一张 CT 为全零图，目标图块最终掩膜逐位相同。固定中心控制包含全队列。
- 原始 E11 基线仍为 `/home/user/yc67201/.local/share/diffoseg-runs/e11_lidc_formal_20261104a`。v9 比较对象的汇总 SHA-256 为 `33226deadde97f57179f0c261b5631fd50c86b8f64e2956a06b56ab56a33c281`。本轮新结果汇总 SHA-256 为 `2024e556c9cf003d037701d458d2da3eb527deb0192bf8666cf8383651a220d0`。
- Lakers 交付目录为 `/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_preference90_comparisonb`。`assessment/preference_report.json` 包含完整四折和分等级结果，`assessment/fold-F/paired.json` 与该目录内的 NPZ 保存逐例评分及预测；`independent_audit.json` 和 `single_slice_control.json` 均在 `assessment/`。锁定参数保存在运行根目录的 `lock-F.json` 和 `selection.json`，单切片前向检查另见 `center-audit.log`。全套 58 项测试通过。
- 配准模型、放大模型、面积估计器和定位器分别位于 `/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_single_slice_90_aligned`、`/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_single_slice_90_zoom`、`/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_single_slice_90_area`、`/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_single_slice_90_native_v10c`。折 0、2、3 配准/放大模型在 Lakers 训练，折 1 在 E11；四个全折定位器在 E11 分配的 GPU 上训练。N21 同步源码。E11 的 230045、230046 是失败 smoke，230047–230054 成功完成。早期 smoke 修正了 OpenCV 导入和层名冲突；内层汇总还修正了一处将数值分数当作哈希的检查。失败记录均保留。三端主要运行源码清单为 `49786de181fd5bfccfd365adee6cf7eeffd9f0ce72b8c20ecaefd8d80bfc3276`；添加独立输入核验程序后的完整交付源码清单均为 `a795bb667aca7c09ab352abbc73faf726fa8fbe9b76069c2c4fcb0c8a3e04cb8`。E11/N21 对应各自 `$HOME/.local/share/diffoseg-runs/20261106_preference90_comparisonb/release-source`。
- 训练入口为 `scripts/lidc_native_local.sh`、`scripts/lidc_aligned_train.sh`、`scripts/lidc_registered_zoom.sh` 及 `reproduction.rater_area`。最终入口为 `reproduction.single90_study` 的 calibrate、gate、infer、score、report，按阶段完成所有折后再进入下一阶段。额外的纯单片依赖与固定中心核验入口为 `reproduction.native_center_audit`。原有冻结代码和权重是重放依赖。[初始协议](protocol.md) · [配准方案](alignment_protocol.md) · [高分辨率方案](zoom_protocol.md) · [面积估计](area_protocol.md) · [选择规则](decision.md)

```bash
R=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_preference90_comparisonb
cd "$R/release-source"
sha256sum --quiet -c ../release-source.sha256
python3 -m json.tool "$R/assessment/independent_audit.json"
python3 -m json.tool "$R/assessment/single_slice_control.json"
```
