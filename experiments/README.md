# 实验入口

统一使用 `python -m experiments.run --config <path>`，而非为每个实验复制推理代码。

| 实验 | 配置 | 目的 |
| --- | --- | --- |
| Baseline | `configs/experiments/baseline.json` | 原图 + 问题 + 选项 |
| Text-only | `configs/experiments/text_only.json` | 与 baseline 同一批样本；请求中不传图像 |
| Evidence masking | `configs/experiments/evidence_masking.json` | 原图、证据遮挡与等尺寸对照遮挡配对比较 |
| Ablation | `configs/experiments/ablation.json` | 多档模糊强度和降采样比例 |

`resolution` 先降采样再恢复原尺寸，测量细节损失。`blur` 对所有图片施加高斯模糊。所有条件保留相同问题和选项顺序。新增条件只需增加一个配置项；新增变换需扩展 `src/perturbations/transforms.py` 并补充测试。

每次结果目录不可覆盖。中断后换 `--output-dir` 重新运行会自动复用已写入缓存的预测；失败调用不会伪装成弃答。已经得到但解析失败的原始响应会保留，作为 invalid 计入分母。
