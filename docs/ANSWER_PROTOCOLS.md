# P0-4：双作答协议

协议实现位于 `src/models/protocols.py`，通过 `configs/protocols/` 选择。普通实验每次只运行一种协议，避免把不同作答规则混成一个准确率。模型、视觉预处理、题目与选项保持一致，协议只改变作答规则。

| 协议 | 版本 / ID | 行为 |
| --- | --- | --- |
| 强制作答 | `1.0` / `forced_choice@1.0` | 即使不确定也必须给出一个合法选项，`abstain=false` |
| 允许弃答 | `1.0` / `allow_abstain@1.0` | 可选择答案；认为证据不足时返回 `answer=null, confidence=null, abstain=true` |

两者都返回 `answer/confidence/abstain` 三个字段。作答时 confidence 可为 [0,1] 的数或 null；它是模型自报值，不是经过校准的正确概率。允许弃答不等于输入具有不可回答性真值，不能把所有无图/退化题都当成“应该弃答”。

## 版本与摘要

- `protocol_id`：协议名与版本的组合；另保存 `name/version/parser_version`。
- `template_sha256`：尚未填入问题与选项的模板，其原始 UTF-8 文本的 SHA256。
- `prompt_sha256`：填入实际问题与选项后，完整原始 UTF-8 prompt 的 SHA256；不是 JSON 字符串编码后的 hash，不做换行或 Unicode 归一化。
- 本地模型的 `telemetry.rendered_prompt_sha256`：套用 checkpoint chat template 后、展开 image placeholder 前的字符串 hash。图片数量影响该文本；实际视觉 token 数另外由 P0-3 记录。

发布的 v1.0 模板不得直接修改；改动应注册新版本、更新解析兼容性与测试。测试锁定了两个模板的摘要，防止无意改变既有版本。未提供协议的旧调用默认解析为 `allow_abstain@1.0`；普通运行器会把解析后的配置写入 resolved config，不再依赖隐式默认。未知协议或未知版本在推理前报错。

v1.0 的 allow_abstain 文本保持原有提示词内容，但新解析器比历史实现更严格：拒绝重复/额外键、非有限 JSON 数值和非 null 的弃答 confidence。旧结果不自动重新标记为 v1.0 实验。

## 输出解析与统计

解析器版本为 `strict-json-1.0`。允许单个完整 JSON 对象或一个完整的 json/无语言 Markdown fence；选项字母接受大小写及两侧空白。拒绝重复键、额外字段、越界选项、NaN/Infinity、非布尔 abstain 等；不从自由文本猜测答案，也不自动修复后重试。

| 模型输出 | forced_choice | allow_abstain |
| --- | --- | --- |
| 合法选项、合法/null confidence、abstain=false | 有效作答 | 有效作答 |
| answer=null、confidence=null、abstain=true | 协议无效，`valid=false, abstain=false` | 有效主动弃答 |
| answer=null、abstain=false | 无效 | 无效 |
| 自由文本拒答、无效 JSON、越界选项 | 无效，不计主动弃答 | 无效，不计主动弃答 |
| OOM、网络/模型执行异常 | 执行失败 | 执行失败 |

预测记录保留原始 `raw`、`error` 和 `error_kind`（`format` 或 `protocol`）。错误输出不从样本分母删除。准确率以全部题为分母，有效弃答和无效输出均不算正确；弃答率只计协议允许且解析有效的主动弃答。指标另外报告 `n_protocol_violations` 和 `n_format_errors`。执行错误沿用失败 manifest/日志，不伪装成弃答，也不作为已完成实验发布。

## 缓存隔离

统一缓存 helper 由普通运行器和真实双协议 smoke 共用。`prediction-cache-2` 标识包含：协议名/版本、parser 版本、模板 hash、完整 prompt/hash、图片顺序 hash、模型与推理配置、样本/种子、选项数、processor fingerprint 和代码摘要。

读取时核对整个 identity 与 key，内容不一致时报错。即使人为让两种协议的 prompt 字节完全相同，协议字段仍使它们使用不同缓存。旧缓存不直接迁移；相同版本与完整 identity 才可重用。

缓存保存原始响应与 telemetry，解析在当前已锁定的协议下执行；缓存命中记录的是原始调用的耗时和显存，不意味着本次重新使用 GPU。完整问题文本保存在缓存 identity，prompt hash 保存在预测行。

## 运行命令

普通运行器支持协议覆盖（以下为开发集运行示例，不是 50 题分层 pilot）：

```powershell
.\scripts\python.ps1 -m experiments.run --config configs/experiments/development.json --model-config configs/models/local_4070.json --protocol-config configs/protocols/forced_choice.json --limit 6 --output-dir outputs/dev-forced-v1
.\scripts\python.ps1 -m experiments.run --config configs/experiments/development.json --model-config configs/models/local_4070.json --protocol-config configs/protocols/allow_abstain.json --limit 6 --output-dir outputs/dev-abstain-v1
```

本阶段配对 GPU 验收使用以下专用入口，确定性选择同一批开发题、只加载一次模型：

```powershell
.\scripts\python.ps1 -m experiments.protocol_smoke --samples 6 --output outputs/smoke/p0-4-protocols-2026-10-29
.\scripts\python.ps1 scripts/summarize_protocols.py --manifest outputs/smoke/p0-4-protocols-2026-10-29/manifest.json --output docs/reports/p0-4-2026-10-29.json
.\scripts\python.ps1 -m unittest discover -s tests -v
```

6 题 × 原图/Text-only × 两协议 = 24 次冷启动请求；两个协议共用同一个新缓存目录，随后进行 24 次只读缓存重放。汇总器核对开发边界、完整配对、相同图像/网格/视觉 token、协议/prompt hash、缓存 key 隔离以及原始响应和 telemetry 一致性。运行目录不可覆盖。

真实模型可能在允许弃答下仍全部作答。不能为了展示弃答而按结果挑题或修改样本；解析差异由固定响应的单元测试验证，行为差异留给后续配对实验估计。
