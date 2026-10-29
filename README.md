# microscopy-evidence-reasoning

面向显微镜多模态推理研究的可复现实验框架。以 **MicroVQA** 为 benchmark，研究模型的回答是否依赖视觉证据，支持数据处理、VLM 推理、证据干预、配对评估和论文作图。

项目提供离线演示、自动化测试和本机真实模型验证。**mock 只用于测试基础设施，不代表模型能力；真实 smoke 已完成，正式 MicroVQA benchmark 结果尚未产出。**

## 研究与开发规划

项目按课程/学位研究推进，proposal 截止为 **2026-11-24**。结合本机 4070 Laptop 8 GB、低预算和暂无专家标注的条件，当前主线为视觉输入缺失/退化对准确率与置信度的影响；人工关键区域实验保留为扩展。

- [研究问题、实验矩阵与统计方案](docs/RESEARCH_PLAN.md)
- [开发优先级、验收标准与日期安排](docs/DEVELOPMENT_ROADMAP.md)
- [Proposal 写作结构与最终交付清单](docs/PROPOSAL_AND_DELIVERY.md)

规划文档区分已有功能和待开发能力；下文为目前框架的实际使用说明。

本机真实模型接入、固定版本数据下载、全量审计与分组命令见 [P0 本机操作说明](docs/P0_SETUP.md)。

**2026-10-28 验证进展：**本机 4070 已完成 12 次 Qwen2.5-VL-3B NF4 真实调用；MicroVQA 1042 题通过审计，划分为开发集 206 题与测试集 836 题。详见 [P0 实测报告](docs/reports/p0-2026-10-28.md)。这是可行性验证，尚非正式 benchmark 结果。

**2026-10-29 P0-3：**预处理已显式冻结；6 道开发题 × 四条件共 24 次真实调用通过，实际网格、视觉 token、顺序张量校验及分阶段资源记录齐全。见 [验收报告](docs/reports/p0-3-2026-10-29.md) 和 [预处理配置/复现说明](docs/PREPROCESSING.md)。

**2026-10-29 P0-4：**双作答协议 v1.0 已实现，固定协议与 prompt hash，并隔离缓存。本机 24 次配对调用和 24 次缓存重放通过，49 项测试通过。见 [双协议验收报告](docs/reports/p0-4-2026-10-29.md) 和 [协议说明/运行命令](docs/ANSWER_PROTOCOLS.md)。

## 目录结构

```text
configs/                 模型、数据路径、推理参数与实验条件（JSON）
  models/                mock / API / 本地 VLM
  experiments/           baseline / text_only / evidence_masking / ablation
src/
  data/                  MicroVQA 导出、读取、格式统一与校验
  models/                统一请求接口、适配器、提示词与输出解析
  perturbations/         证据遮挡、对照遮挡、模糊、分辨率变化
  evaluation/            指标、配对统计、CSV 表格与 PDF/PNG 图表
experiments/             统一实验运行入口
tests/                   数据、图像、模型协议、指标与端到端测试
data/                    本地数据（默认不提交）
outputs/                 预测缓存、结果、表格与图表（默认不提交）
pyproject.toml           安装配置和可选依赖
.github/workflows/       Windows/Linux 测试配置
```

## 安装

Python **3.10+**。在项目根目录创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。
也可直接调用 `.venv` 内的 Python，无需改变 PowerShell 执行策略。

```bash
python -m pip install -e .
# 按需安装：官方数据下载、作图、本地 VLM
python -m pip install -e ".[data,plots]"
python -m pip install -e ".[local]"
```

基础依赖只有 Pillow 和 NumPy；测试使用标准库 unittest。API 适配器使用标准库 HTTP 客户端。本地模型可选依赖固定 Transformers 4.57.1，CUDA/PyTorch 安装应按机器环境选择。大型模型和官方数据不会在安装或测试时自动下载。

## 五分钟离线演示

以下命令均在项目根目录执行，使用合成图片和确定性 mock，无需 API 密钥、GPU 或官方数据：

```bash
python -m src.data.prepare demo
python -m experiments.run --config configs/demo.json
python -m src.evaluation.report --run-dir outputs/demo --tables-only
python -m unittest discover -s tests -v
```

安装 `plots` 后导出图表：

```bash
python -m src.evaluation.report --run-dir outputs/demo
```

输出包含 `summary.csv`、`paired.csv`、矢量 `accuracy.pdf` 和 300 dpi `accuracy.png`。图中会标注演示性质。
演示数据和结果目录已存在时不会覆盖；验证缓存复用可执行：

```bash
python -m experiments.run --config configs/demo.json --output-dir outputs/demo_replay
```

## 准备 MicroVQA

官方来源：[数据集与字段说明](https://huggingface.co/datasets/jmhb/microvqa)、[作者代码仓库](https://github.com/jmhb0/microvqa)、[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Burgess_MicroVQA_A_Multimodal_Reasoning_Benchmark_for_Microscopy-Based_Scientific_Research_CVPR_2025_paper.html)。

当前 Hugging Face 数据卡使用 **test** 划分，主要字段为 `key_question`、`key_image`、`images_list`、`question`、`choices`、`correct_index`。部分问题有多张图片，本框架保留它们的顺序，不做隐式拼图。作者旧版 eval README 中出现过 train，实际接入以固定版本的数据卡为准。

在 Hugging Face 的 Files and versions 中选择数据集 commit，将下列占位符替换为完整 SHA：

```bash
python -m src.data.prepare microvqa --revision DATASET_COMMIT_SHA --split test --output data/microvqa
```

导出 `samples.jsonl`、`images/*.png` 和 `source.json`；revision 会解析为实际 commit 并记录。可加 `--limit 10` 先验证小样本，但这不是完整 benchmark。标签为零起始索引，所有图片会转为 RGB PNG。原始数据许可证与使用限制以官方数据卡为准；benchmark 不应混入训练语料。

统一格式、区域坐标约定和标注要求见 [data/README.md](data/README.md)。关键区域标注需自行提供，首版支持每张图一个矩形；没有标注时可运行 baseline、text-only 与模糊/分辨率消融。

## 运行科研实验

先编辑 `configs/models/api.json` 的服务地址、模型名和认证选项，或 `configs/models/local.json` 的 checkpoint、revision 和设备配置。示例 API 使用兼容 `/v1/chat/completions` 的视觉接口。需要认证时，PowerShell 用 `$env:VLM_API_KEY="..."`，bash 用 `export VLM_API_KEY="..."`；不要把密钥写进配置或提交到仓库。

```bash
# 原图基线
python -m experiments.run --config configs/experiments/baseline.json --model-config configs/models/api.json

# 原图与无图在同一批样本上配对
python -m experiments.run --config configs/experiments/text_only.json --model-config configs/models/api.json

# 先准备 data/microvqa/samples.annotated.jsonl
python -m experiments.run --config configs/experiments/evidence_masking.json --model-config configs/models/api.json

# 多档高斯模糊与降采样；也演示本地模型适配器
python -m experiments.run --config configs/experiments/ablation.json --model-config configs/models/local.json

python -m src.evaluation.report --run-dir outputs/evidence_masking
```

可用 `--limit 10`、`--data-path ...`、`--output-dir ...` 覆盖配置。配置继承规则见 [configs/README.md](configs/README.md)。默认配置使用 mock，正式实验务必选择真实模型并检查 manifest 中的 `is_mock=false`。安装后也可使用 `microvqa-prepare`、`microvqa-run`、`microvqa-report` 命令。

### 实验条件

| 条件 | 实际输入与变换 |
| --- | --- |
| Baseline | 原图 + 同一问题和选项 |
| Text-only | 同一问题和选项；请求中无图像块 |
| Evidence masking | 将人工标注矩形填充为配置颜色 |
| Control masking | 同图内随机选择等形状、等面积且与证据框不相交的矩形 |
| Blur | 对所有图片施加指定 radius 的高斯模糊 |
| Resolution | 按 scale 降采样后恢复原尺寸，测试细节损失 |

区域缺失或无法放置对照框时，在调用模型前终止，不静默删除样本。模型仅接收问题、选项、干预后的图片及生成 seed，不接收标签、标注或数据元信息。对照遮挡仅保证几何匹配，不能保证对照区域在语义上无关；需在研究设计中审查。

### 输出与指标

模型输出约定：`{"answer":"A","confidence":0.8,"abstain":false}`。
主动弃答：`{"answer":null,"confidence":null,"abstain":true}`。支持完整 JSON 代码围栏；自由文本与越界选项记为解析失败，不从回答中猜测字母。

| 指标 | 定义 |
| --- | --- |
| accuracy | 正确回答数 / 全部样本数；弃答与解析失败计为未答对 |
| coverage | 有效且未弃答数 / 全部样本数 |
| abstention_rate | 有效主动弃答数 / 全部样本数 |
| invalid_rate | 解析失败数 / 全部样本数；与主动弃答区分 |
| selective_accuracy / risk | 有效作答子集上的准确率 / 错误率；无人作答时为 null |
| mean_confidence / ECE | 仅对有效作答且提供 confidence 的样本计算；缺失时为 null |
| correctness_brier | 自报置信度与答对指示变量的均方差；不是多分类 Brier score |

confidence 是模型自报信心，尚不提取 token log-probabilities。指标同时保存 `n`、`n_answered`、`n_confidence`，避免分母混淆。API/模型执行失败将整次运行标记 failed，不伪装为弃答。

配对比较严格按样本 ID 对齐，要求集合和标签一致。`accuracy_delta = intervention - baseline`；95% CI 采用 percentile bootstrap，有 `key_image` 时按图像组重采样，无分组时按问题重采样。McNemar 为双侧精确检验，其独立问题对假设不处理共享图像相关性；多条件的多重比较校正尚未实现。图像组很少的演示 CI 不具有研究解释价值。

## 复现与扩展

1. 固定代码 commit、数据集 commit、区域标注版本、模型 checkpoint 和推理配置，记录硬件与服务端版本。
2. 使用相同样本集比较所有条件。固定 seed 会固定控制框、bootstrap 和支持 seed 的模型采样；不保证跨硬件、CUDA 实现或远程服务完全逐位相同。
3. 每次运行保留 `config.resolved.json` 与 `manifest.json`，其中记录数据/图片/源码 SHA256、Python/包版本、Git 状态、样本列表和缓存命中数。
4. 原始响应逐条写入预测文件，缓存原子写入。缓存键含实际输入图像内容、提示词、模型参数、seed 与代码摘要。相同请求可跨运行复用；变更代码会使旧缓存失效。
5. 中断后指定新结果目录重跑，自动复用已完成预测。只分析 `manifest.status=complete` 的运行。API 同名模型更新、本地 checkpoint 内容变更时应更换 `cache_dir` 或固定版本名称。
6. 正式实验建议从干净虚拟环境安装并保存完整依赖锁定文件：`python -m pip freeze > outputs/environment.lock.txt`。`pyproject.toml` 是兼容范围，不是跨平台完全锁定的科研环境。

新增模型实现 `generate(ModelRequest) -> str` 并注册到 `src/models/factory.py`；本地适配器目前面向支持多模态 chat template 的 decoder-only Transformers VLM，仍需针对所选模型进行实机验证。API 适配器支持有限次数的网络/429/5xx 重试，不宣称所有服务完全兼容。

`tests/` 覆盖数据校验、多图顺序、精确遮挡面积、控制框不重叠、图像不被原地修改、解析/弃答、指标、配对统计、缓存与失败恢复。CI 配置为 Windows/Linux × Python 3.10/3.12；真实 GPU 推理和外部 API 不属于离线测试范围。

更详细的产物说明见 [outputs/README.md](outputs/README.md)。该代码框架实现本文档描述的实验协议，并非对原论文全部提示词、模型设置和结果的复刻。
