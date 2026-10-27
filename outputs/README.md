# 实验产物（默认不提交）

每次运行包含：

- `config.resolved.json`：继承与 CLI 覆盖后的配置。
- `manifest.json`：状态、时间、种子、Git 状态、源代码/数据/图片 SHA256、包版本、样本列表、数据来源、缓存命中计数。
- `predictions.jsonl`：样本、条件、原始响应、解析结果、标签、缓存键、实际干预区域。
- `metrics.json`：每个条件的准确率、覆盖率、弃答、解析失败和置信度指标。
- `paired.json`：相对 baseline 的配对差值、bootstrap CI 和 McNemar 检验。
- 报告命令生成 `summary.csv`、`paired.csv`、`accuracy.pdf`、`accuracy.png`。

`outputs/cache/` 保存原始模型响应。缓存键包含模型配置、推理参数、样本 seed、问题提示、实际图像内容及源码摘要。不要手工编辑缓存；需要重新调用相同请求时使用新的 `cache_dir`。服务端模型在同名下变更时必须换版本标识或缓存目录。

只有 `manifest.status=complete` 的结果可用于报告。mock 图表会显式标注为演示，不能用于论文结论。论文归档时应另行保存配置、环境锁定文件、标注版本、数据来源 commit 和所需预测；选择性提交体积小的表格时可使用 `git add -f`。
