# 本机 P0 环境、数据审计与 Smoke Test

适用硬件：RTX 4070 Laptop 8 GB。以下命令在项目根目录的 PowerShell 执行。完整执行结果另见 `docs/reports/`；本文件描述可复现流程。

## 1. 环境

当前选用 CUDA 12.4 构建的 PyTorch 2.6.0 + torchvision 0.21.0，与本机 555.97 驱动配合验证。具体依赖见 [local-4070.txt](../requirements/local-4070.txt)。首次安装时先创建 `.venv`，再安装 GPU wheel，避免误装 CPU 版。

```powershell
python -m venv .venv
.\scripts\python.ps1 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\scripts\python.ps1 -m pip install -r requirements/local-4070.txt
.\scripts\python.ps1 -m pip install -e ".[plots]"
.\scripts\python.ps1 -m pip check
```

`scripts/python.ps1` 使用项目虚拟环境，将 HF 缓存放在 `.cache/huggingface`，将临时目录放在 `outputs/tmp`，启用 UTF-8，并取消继承机器上的 TLS key-log 设置；不修改系统级环境变量。缓存、权重、图片不提交 Git。

本次开发环境的 `.venv` 继承了桌面运行时已有的基础包；最终科研归档应使用上述干净环境复核，并保存完整 freeze 记录。不要将本机有 CUDA 张量运算等同于大模型输入路径已经验证。

## 2. 下载固定数据与模型

固定版本见 [assets.json](../configs/assets.json)。下载脚本只访问公开官方仓库，不加载远端 Python 代码；下载失败保留已有缓存，可以重试同一版本。

```powershell
.\scripts\python.ps1 scripts/download_assets.py --kind dataset --revision 744d656b41f2d9f9bf8f229283271bd67ee782fd --manifest outputs/setup/dataset-download.json
.\scripts\python.ps1 scripts/download_assets.py --kind model --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 --manifest outputs/setup/model-download.json
```

首次下载体积较大。`hf_xet` 用于官方大文件传输。后续导出可直接读取已下载的 snapshot，无需重新访问 Hub。

```powershell
.\scripts\python.ps1 -m src.data.prepare microvqa --revision 744d656b41f2d9f9bf8f229283271bd67ee782fd --download-manifest outputs/setup/dataset-download.json --output data/microvqa
```

只读取该版本的 `data/test-*.parquet`；仓库中历史 train 文件不作为本研究训练集，也不混入导出。已有数据目录不会覆盖。新导出写入完成状态与临时 JSONL，失败不会留下可误认为完整的正式样本文件。

## 3. 审计与分组划分

```powershell
.\scripts\python.ps1 -m src.data.audit --data data/microvqa/samples.jsonl --output outputs/audit/microvqa-v1.json
.\scripts\python.ps1 -m src.data.splits --audit outputs/audit/microvqa-v1.json --output configs/splits/microvqa-v1.json --dev-fraction 0.2 --seed 42
```

审计会检查所有 JSONL 行、答案索引及文本一致性、图片可读性，统计任务/图片数量/尺寸/选项数量，并输出全部错误。错误不被静默过滤；无效审计不能产生划分。

分组规则：共享 `key_image` **或**共享完全相同的解码 RGB 图像的题目合并成连通分量，包括传递关系。识别范围为精确图片重复，不保证识别近似重复、裁剪图或来自同一原始标本的不同图像。

划分按合并后的图像组进行，目标约 20% 组用于开发。在固定种子下生成 512 个候选，按总题数和“任务类别 × 单/多图”比例选择较平衡的候选；不使用答案正确性或模型输出。图像组不可拆开，因此题数比例只近似 20%。

清单保存开发/测试 ID、合并后的 group ID、数据 SHA256、图片文件 SHA256、分组算法、种子和自身摘要。该清单应随代码版本保存；审计详细文件保留在本地输出目录。数据或图片变更时，运行器会拒绝旧清单，不能重用旧 split 悄悄开展新实验。

`configs/experiments/development.json` 展示正式运行器如何使用 `split_manifest` 与 `partition`。运行器采用合并后的组进行配对统计；`limit` 在选定分区后生效，仍只是取前 N 个，不能替代分层抽样或 smoke 选择策略。

## 4. 本机真实模型 Smoke Test

```powershell
.\scripts\python.ps1 -m experiments.smoke --model-config configs/models/local_4070.json --output outputs/smoke/qwen25vl3b_nf4_4070
```

模型配置：固定 Qwen2.5-VL-3B commit、NF4 双重量化、SDPA、单 GPU、单题推理、每图受控像素范围。模型与处理器仅使用本地缓存，不发生测试期间自动换模型或下载新版本。具体 dtype 和像素数以配置及 manifest 为准。

Smoke 默认从开发分区确定性选择 6 题：2 道单图、2 道多图、开发集图片数量最多的一题，再按种子补足去重后的数量。每题运行原图和 Text-only；多图题保留全部图片，Text-only 不携带图片。不会在测试分区上做预测。

产物 `manifest.json` 记录：

- Python、torch、CUDA、模型/量化库版本、显卡和驱动。
- 模型加载时间、解析出的模型 revision、device map、模型内存和处理器设置。
- 开发样本 ID、split ID、输入图片数量与顺序哈希、实际 `image_grid_thw`、输入/生成 token 数。
- 每个请求耗时、峰值 allocated/reserved 显存、原始响应、JSON 解析结果、失败类型。

`status=complete` 表示所有真实调用执行完成；`n_valid_json` 另外表示协议解析后的有效程度。答错和有效弃答不是基础设施失败。Smoke 默认使用 allow_abstain@1.0，也支持 `--protocol-config`；双协议配对验证见 [协议说明](ANSWER_PROTOCOLS.md)。

这是链路可行性验证，不是 benchmark 得分，也不代表所有 1042 道题都已通过显存压力测试。运行目录不能覆盖；重测需换新目录，避免混合配置结果。

## 5. 验证与下一步

```powershell
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 -m pip freeze > outputs/environment-p0.lock.txt
```

P0-3 的固定 processor、输入预算、视觉 token 和运行记录见 [预处理冻结说明](PREPROCESSING.md)。P0-4 双作答协议已完成；后续固定 P0-5 样本清单与完成 P0-6 输入预览，再扩大到约 50 开发题的四条件 pilot。不要因为 smoke 可运行就直接在锁定测试集上调参。
