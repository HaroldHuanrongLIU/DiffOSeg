# CISC8006 课程项目交付入口

文档更新日期：2026-11-11。当前为 **submission candidate**，职责分工、用户报告的教师确认及冻结信息见 [release_status.md](release_status.md)。本文不改变已提交的 claim。

## 先看结论

- **原 claim：** DiffOSeg Table 3 的 Stage I P/A 单折、各三 seed 比较已经完成，结论为 **inconclusive**。两项均值向预期方向变化，但配对 95% 区间均跨零。
- **改进主报告：** 选择已有的边界校正、极坐标负结果、票数监督和联合解码/能量校准作为完整研究链。它们使用另外冻结的 Reproduced DiffOSeg 四折基线（归档名称 E11）；K=50 是历史主端点，K=30 为保留的次要端点。不能将其改写成原 P/A 比较。
- **补充结果：** Final E 的四折偏好 Dice_mean 为 90.5210%，相对 Reproduced DiffOSeg 为 +2.4985 pp。属于重复使用同一队列的增强系统评估，包含额外模型和计算。

## 阅读和复查

| 内容 | 入口 |
|---|---|
| 提交的原始文本 | [submitted claim](../course_project_requirement/CISC8006-submitted-claim.md) |
| 协议、实现及判定 | [claim_and_protocol.md](claim_and_protocol.md) |
| 差异诊断 | [diagnosis.md](diagnosis.md) |
| 三个以上假设、pilot、完整评估和消融 | [improvements.md](improvements.md) |
| 表格到原始产物 | [evidence_map.md](evidence_map.md) |
| 计算预算及缺口 | [compute.md](compute.md) |
| 重放、环境与数据 | [reproduce.md](reproduce.md) |
| 个人贡献和 AI 使用 | [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md)、[AI_USAGE.md](../../AI_USAGE.md) |
| 实际技术复算记录 | [peer_audit.md](peer_audit.md) |
| 报告和演示 | [report](../../report/course_project.typ)、[slides](../../presentation/course_project.typ)、[答辩准备](defense.md) |

从仓库根目录运行，无需 GPU 或医学图像：

```bash
python3 -B scripts/check_course_project.py
```

预期输出以 `SMOKE_PASS` 开头，并显示六次运行、2700 条记录、54 个患者簇及 `inconclusive`。这是**交付包证据检查**；训练 smoke 和历史模型评估有独立说明。

已安装项目环境时，可复算全部 bootstrap：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/check_course_project.py --bootstrap
```

表格由 JSON 生成，避免手填数值。执行 `python3 -B scripts/render_course_tables.py` 后运行 `bash scripts/build_course_documents.sh` 生成基于 UM_Typst_Template 子模块的英文报告和 slides PDF。

已生成的英文材料：[报告 PDF](../../report/course_project.pdf)、[slides PDF](../../presentation/course_project.pdf)、[slides PPTX](../../presentation/course_project.pptx)。PPTX 是模板规定的逐页图像版，编辑内容请改配套 Typst 源文件。2026-11-11 已按“问题与论文 → claim 复现 → 受控改进 → 证据与局限”重构为含封面的 15 页；讲稿和时间分配见 defense.md，图表再生成及论文配图出处见 presentation/assets/README.md。

报告已同步更新为 7 页英文版，补充 pilot 定量结果与晋级规则、v4 方法与选择过程、偏好性能提升路径，以及每人 5 页／约 5 分钟的演讲安排。报告新增表格直接读取归档 JSON；校验记录见 `results/course_project/report_validation.json`。
