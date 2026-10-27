# 本地数据

本目录只跟踪此说明文件，下载数据与合成数据均被 Git 忽略。

先运行 `python -m src.data.prepare demo`，或通过 `microvqa --revision <commit-sha>` 导出官方数据。详见项目 README。

统一 JSONL 格式（一行一个样本）：

```json
{"id":"644","question":"...","choices":["Choice one","Choice two"],"answer_index":0,"images":["images/00000_00.png","images/00000_01.png"],"evidence_regions":[{"image_index":0,"box":[10,20,50,60]}],"metadata":{"key_image":125,"task_str":"hypothesis_gen"}}
```

- `answer_index` 从 0 开始；选择题选项数 2–26。
- 图片顺序严格保留。输入统一转为 RGB，导出为 PNG；这是实验预处理的一部分。
- 区域使用原始导出图片的像素坐标 `[x0,y0,x1,y1)`，右侧/下侧边界不包含在遮挡内。
- 首版每张图最多标注一个矩形，一个样本可在多张图上标注。尚不支持分割掩膜和同图多个不连续区域。
- 官方数据卡未提供本框架所需的区域标注。研究者需独立标注，并另存 `samples.annotated.jsonl`，记录标注者、方法与版本；不可把合成演示区域当作 MicroVQA 标注。
- 遮挡实验中所有样本必须至少有一个区域。无区域或放不下同尺寸对照框时，整次实验在调用模型前报错，不静默丢弃样本。需要研究者明确构造可用子集，并让所有对比条件使用该子集。
- 不向模型传递正确答案、区域标注或元数据。保留 `key_image` 用于按图像组 bootstrap。
- 官方字段 `key_question`、`images_list`（本地路径列表）、`correct_index` 也可直接读取；HF PIL 图片对象应先通过导出命令落盘。
