# P0-3：冻结预处理与输入审计

本阶段使用 [本机模型配置](../configs/models/local_4070.json)。配置适用于已验证的 Qwen2.5-VL fast processor；通用 `local.json` 示例不提供同等冻结保证。变更像素预算或 processor 参数视为新实验版本，应统一重跑所比较的条件。

## 冻结内容

| 项目 | 配置 |
| --- | --- |
| processor | `Qwen2_5_VLProcessor` |
| image processor | `Qwen2VLImageProcessorFast`，`use_fast=true` |
| 模型、tokenizer、chat template 来源 | 固定模型 commit 的本地 snapshot；禁止自动切到 main |
| 每图面积预算 | 50176–200704 像素；`min_pixels/max_pixels` 与 `size` 同时显式指定 |
| 缩放 | 保持宽高比的 Qwen smart resize，空间尺寸按 28 像素对齐；bicubic，`resample=3` |
| 颜色与归一化 | RGB；乘以 1/255；明确配置三通道 mean/std |
| 模型 patch 参数 | spatial patch=14，temporal patch=2，merge=2 |
| 图像处理位置与顺序 | CPU；`disable_grouping=true`；按照请求列表顺序传入全部图片 |
| 文本 | `padding=false`、`truncation=false`；采用固定 checkpoint 的 chat template |

加载时检查实际类名、参数和 `size`，与配置不符即报错。processor fingerprint 覆盖实际设置、类名、chat template 摘要及 Transformers/PyTorch/torchvision/Pillow 版本。模型 revision、prompt 和图片内容另外参与普通实验缓存标识。

每图预算不是整题预算：多图题保留全部图片，总视觉 token 随图片数量增长。Text-only 不传入图片，不以空白图代替。发生 OOM 时记录错误，不自动减少图数或改变像素预算。

## 每次调用记录

`telemetry` 保存请求样本 ID、图片模式、尺寸与顺序 RGB 哈希、processor fingerprint，以及：

- 实际 `image_grid_thw`，每图处理后 `[height,width]`、面积和 patch 数。
- 每图与整题视觉 token：`t*h*w / merge_size²`。静态图片 `t=1`；复制 temporal patch 帧不会把 LLM token 数翻倍。
- 实际输入中 `<|image_pad|>` 的数量，并校验其与网格推导的总数一致。视觉 token 不包含图像起止标记，也不等于全部 input token。
- 按网格切分的每图实际 `pixel_values` 张量摘要，可用于核对顺序和退化后输入是否改变。
- 预处理、设备传输、生成（含解码）和总耗时；生成 token 与是否达到长度上限。
- 每张 CUDA 卡的峰值 allocated/reserved，以及单请求汇总。

计时包含请求内校验和张量摘要计算，不包含数据集审计、图像文件解码、扰动构建、模型加载及额外的逐张顺序复核。显存峰值从本次调用开始重置，包含仍驻留的模型，不包含其他进程或驱动占用；不代表加载阶段峰值。多卡汇总是各卡峰值之和，未必同时达到；当前验证为单卡。

普通运行器在 `manifest.json` 保存 `model_load`，在 `predictions.jsonl` 和缓存中保存 telemetry。命中缓存时保留原始推理的耗时/显存，不代表本次发生了 GPU 推理；实际命中次数见 manifest。仅凭本次加载参数不能覆盖缓存记录。加载本地模型后才查缓存，因此纯缓存重放仍需本地模型资源；直接从预测重建报告不需要模型。

失败调用在 telemetry 中记录 `status/error_type/error/stage`，普通运行器另外保存 `failure.json`，不会把执行错误转换成主动弃答。此阶段维持首个错误终止的运行器策略，批量自动恢复留待 P1。

## 真实 GPU 验收

在已完成 P0-1/P0-2 的环境中运行；输出目录必须不存在：

```powershell
.\scripts\python.ps1 -m experiments.smoke --model-config configs/models/local_4070.json --conditions-config configs/smoke_preprocessing.json --verify-image-order --samples 6 --output outputs/smoke/p0-3-preprocessing-2026-10-29
.\scripts\python.ps1 scripts/summarize_preprocessing.py --manifest outputs/smoke/p0-3-preprocessing-2026-10-29/manifest.json --output docs/reports/p0-3-2026-10-29.json
.\scripts\python.ps1 -m unittest discover -s tests -v
```

只选择已有开发分区中的 6 题（含最大图数开发题），每题原图、Text-only、模糊与降采样共 24 次调用。`--verify-image-order` 独立处理每张图片，将其网格及张量摘要与多图请求的对应片段比较；这一步发生在推理计时之外。

汇总器检查四条件完整、开发分区边界、同题有图条件网格一致、实际预算和视觉 token 一致、Text-only 零视觉 token、独立顺序复核通过，以及此次 smoke 中退化后的张量确实改变。像素内容改变并不能证明关键生物学证据被破坏；完整视觉预览与退化强度选择仍属于 P0-6。

本阶段沿用原有允许弃答提示词。双作答协议属于 P0-4，不把此次 smoke 当作正式 benchmark 或 50 题 pilot。
