# P0-5 / P0-6：固定开发 pilot 与实际输入审计

P0-3/P0-4 已先提交到本地 Git：`c2a2350`。本阶段继续使用同一台 RTX 4070 Laptop、同一固定 Qwen2.5-VL-3B NF4/BF16 和内部开发/测试分区，不使用测试集表现选择题目或退化参数。

## 1. 固定样本清单

[pilot50-v1.json](../configs/samples/pilot50-v1.json) 固定 50 个题目 ID、对应合并图像组、任务类型、图片数、种子、数据 SHA256、父 split ID 和清单自身 selection ID。

选择规则为任务类型 × 单图/多图的比例分层，以最大余数法把 206 道开发题的分层比例转换为 50 个名额。先纳入开发分区中一题最大图片数题用于资源验证，再优先选择尚未覆盖的图像组，使用 seed=42 的稳定 hash 打破平局。不使用标签、正确性或模型响应。

| 任务 | 单图 | 多图 | 总数 |
| --- | ---: | ---: | ---: |
| experiment_proposal | 7 | 5 | 12 |
| hypothesis_gen | 11 | 8 | 19 |
| perception | 11 | 8 | 19 |
| 合计 | 29 | 21 | 50 |

实际覆盖全部 **45 个开发图像组、90 个图片引用**，最大 6 图题为 ID 778。为了覆盖图像组，选取的是组内部分问题；组内所有问题不必一起纳入 pilot，但父开发/测试分区仍不可拆分或改变。这里 50 题不是 50 个独立图像组，配对 bootstrap 仍使用合并组。

Selection ID：`4f7fda19e599de3942b23c225d5d9ac38be8438bb3ee8a66ce489c49d01cf71f`。

运行器按照清单顺序选题，拒绝重复、缺失、非开发题和不匹配的清单；使用该清单时禁止 `--limit`，防止无意截断成不完整 pilot。每题的所有条件使用同一问题、选项、生成种子和协议；结束前核对完整笛卡尔积，没有以预测正确性删题。

## 2. 输入顺序和四条件

固定处理顺序：**原始导出 RGB → 公共研究尺寸 → 四条件干预 → 冻结模型 processor → 模型**。

公共研究尺寸使用与 Transformers 4.57.1 Qwen smart resize 相同的面积/28 像素对齐规则，面积 50176–200704；图像重采样明确使用 Pillow bicubic。此公共步骤对所有条件相同；随后 processor 对这套对齐尺寸无需再次改变空间分辨率。

| 条件 | 操作 |
| --- | --- |
| baseline | 公共研究尺寸下的 RGB，不施加退化 |
| text_only | 同一问题和选项，零图片 |
| blur | 每图 Gaussian blur radius = 公共研究图较短边 × 0.02 |
| resolution | 每图宽高降至 25%（LANCZOS），再以 bicubic 恢复公共研究尺寸 |

配置见 [pilot.json](../configs/experiments/pilot.json)。默认强制作答 v1.0；[pilot_allow_abstain.json](../configs/experiments/pilot_allow_abstain.json) 使用相同清单和视觉设置，只切换协议与输出目录。

**本阶段 baseline 增加了显式 Pillow 公共缩放，不能与此前直接处理原图的 smoke 结果混在一起比较。** 模糊相对强度 0.02、分辨率比例 0.25 是这次 pilot 的工程设置，不表示两者具有等价的视觉或生物学难度；后续修改强度应生成新版本配置和审计。

## 3. 实际 processor 输入审计

审计仅加载固定 processor，不加载模型权重。对 50 题的全部图像逐条件处理，并保存：

- 原始尺寸/哈希、研究尺寸/哈希、最终条件图片哈希、实际网格、patch 数与推导视觉 token。
- 按网格切分的每图实际 `pixel_values` 张量 SHA256。
- 从 Qwen 空间/时间 patch 排列反向恢复的 RGB 浮点图，反归一化后输出实际大小 PNG。
- 最终退化图对 baseline 的 RGB MAE、平均空间梯度比和张量是否改变。

预览的“Actual tensor”来自实际张量反向重建，不是另行模拟的变换图。审计逐张要求重建 uint8 RGB 与送入 processor 的干预图完全一致，且尺寸不变。这验证后续处理没有重采样或抵消本次干预。PNG 反归一化后经过 uint8 舍入；数值差异统计在 RGB 浮点值上计算。

此版本审计检查所有 90 张图片：原图/模糊/降采样同题网格相同，Text-only 为零图，180 个退化输入的张量均与对应 baseline 不同。任何完全相同的退化张量被记录为失败；不能静默删掉样本再声称通过。

每个条件共用同一面积预算，但同题多张图片的尺寸可能不同；模型顺序始终为原始列表顺序。实际生成时，运行器还会把最终输入哈希、网格和张量哈希逐项与审计比较。审计未通过、配置不匹配、processor fingerprint 改变或实际输入不一致时会停止，不发布完整实验。

“张量改变”只证明操作存留，不证明关键生物学证据损失。像素变化很小的图也要保留报告；全图 MAE 会被大面积暗背景稀释，不单独作为科学有效性阈值。ID 778 的第 4 张图属于此类例子。

## 4. 本地预览与复现

以下命令在项目根目录运行。固定清单已入仓库，第一条只供重新生成到新路径，不要覆盖现有锁定文件。

```powershell
.\scripts\python.ps1 -m src.data.pilot --output configs/samples/pilot50-v1.json --samples 50 --seed 42
.\scripts\python.ps1 -m src.evaluation.input_audit --config configs/experiments/pilot.json --output outputs/input-audit/pilot50-v1
.\scripts\python.ps1 scripts/plot_input_diagnostics.py --directory outputs/input-audit/pilot50-v1
.\scripts\python.ps1 scripts/build_input_gallery.py --directory outputs/input-audit/pilot50-v1
.\scripts\python.ps1 -m experiments.run --config configs/experiments/pilot.json
.\scripts\python.ps1 scripts/summarize_pilot.py --output docs/reports/p0-5-p0-6-2026-10-29.json
.\scripts\python.ps1 -m unittest discover -s tests -v
```

已存在输出目录会被拒绝，重跑要另选目录并同步 `input_audit` 路径。模型与数据均使用此前下载的固定本地资产，无需新增下载或租机。审计默认使用 CPU；推理默认使用本机 CUDA，单题处理、最多 128 个生成 token。

产物目录：

```text
outputs/input-audit/pilot50-v1/
  audit.json              全部图像与四条件的张量/尺寸/退化记录
  index.html              按题展开的本地预览
  images/                 原图缩略图、研究图、实际输入 PNG 和对照拼图
outputs/pilot50-v1/forced_choice/
  manifest.json           版本、样本、协议、完整性与资源记录
  predictions.jsonl       200 个配对位置的原始/解析输出和 telemetry
  metrics.json            开发 pilot 汇总，不是锁定测试成绩
  paired.json             按合并图像组的配对分析
```

预览拼图为适应屏幕而缩放；单独 `*-baseline.png`、`*-blur.png`、`*-resolution.png` 保留实际 processor 尺寸。Text-only 没有“空白替代图”。预览不显示标签或模型结果，以免通过结果反向挑选展示例子。

大型预测、缓存和图片默认仍由 Git 忽略。仓库保存固定配置、清单和小型验收摘要；实际输出应另行备份。允许弃答的 200 次正式 pilot 可用对应配置另行运行，本阶段优先完成强制作答矩阵。完整论文图表属于后续 P0-7。
