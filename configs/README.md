# 实验配置

配置使用标准 JSON，无需 YAML 依赖。`extends` 相对于当前配置文件解析；支持单个父配置或父配置列表，后者覆盖前者。字典递归合并，数组整体替换。数据、缓存与结果路径相对于项目根目录（CLI 可用 `--root` 指定），图片路径默认相对于 JSONL 文件。

- `base.json`：随机种子、数据路径、推理参数、统计参数。
- `demo.json`：合成数据上的完整离线演示。
- `models/`：mock、兼容 chat/completions 的 API、本地 Transformers VLM。
- `experiments/`：baseline、text_only、evidence_masking、ablation。

实验配置默认使用 **mock**，真实实验必须提供 `--model-config`。API 配置中的模型名是占位符；本地配置是待实机验证的适配器示例，需要自行准备算力。正式实验将 `revision` 改成固定模型 commit SHA 或使用固定本地 checkpoint；API 服务应使用有版本的模型名并保留服务端版本。

关键字段：`data.limit` 为正整数或 null；`temperature=0` 走贪心解码；`seed` 用于干预、bootstrap 和支持 seed 的模型；`ece_bins` 为等宽校准分箱数；`bootstrap_samples` 为重采样次数。API 的 `send_seed` 只应在服务端支持时开启；自报 confidence 不代表 token 概率。

`require_api_key=false` 仅方便本机无认证服务；远程服务按要求设置为 true，并通过配置中 `api_key_env` 指定的环境变量传入密钥。配置、命令行和 Git 中均不写密钥。
