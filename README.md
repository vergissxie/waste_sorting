# Waste Sorting

基于图像分类的 40 类垃圾分类项目。当前正式发布版本为 V2，稳定代码集中在 `src/`，`temp/` 仅保留为历史参考脚本。

## 版本状态

- **V1**：单模型 ConvNeXt Tiny 基线，平台分数 `91.75`
- **V2**：当前正式发布版，双模型融合主线，平台分数 `92.5`

## V2 发布基线

- 主模型：`ConvNeXt Tiny`
- 第二模型：`EfficientNetV2-S`
- 训练方式：`5-fold`、`Weighted CrossEntropy`、`AdamW`、`warmup + cosine`、`AMP`
- 输入尺寸：`256`
- 推理方式：水平翻转 TTA + 多尺度 `256,288`
- 融合权重：`0.6 / 0.4`
- 当前最佳提交结果：`result_e9_c6_e4_256_288.txt`（本地产物，不提交到仓库）

当前发布版不纳入伪标签主线。第一轮伪标签实验 `pl1` 平台分数为 `91.75`，低于当前 `92.5` 融合基线。

## 项目结构

```text
waste_sorting/
  data/                  # 本地数据集
  docs/                  # 方案文档、结构说明、发布检查
  outputs/               # 权重、日志、预测结果
  scripts/               # 辅助脚本
  src/                   # 当前正式源码
    config.py
    dataset.py
    model.py
    train.py
    infer.py
    ensemble_infer.py
    validate_submission.py
  temp/                  # 历史参考脚本
  requirements.txt
```

详细说明见 [docs/项目结构说明.md](docs/项目结构说明.md)。

## 环境安装

建议使用 Python 3.9 或 3.10：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，请按本机 CUDA 版本安装匹配的 `torch` 与 `torchvision`。

## 数据目录约定

```text
data/
  train/
    0/
    1/
    ...
    39/
  test/
  garbage_dict.json
  testpath.txt
```

- 训练集目录名直接对应标签 `0..39`
- 推理输出必须严格按 `data/testpath.txt` 顺序生成
- `result.txt` 必须恰好 `400` 行

## 常用命令

训练 ConvNeXt Tiny：

```powershell
.\.venv\Scripts\python.exe src\train.py
```

训练 EfficientNetV2-S：

```powershell
.\.venv\Scripts\python.exe src\train.py --model-name efficientnet_v2_s --checkpoint-prefix efficientnet_v2_s
```

单模型推理：

```powershell
.\.venv\Scripts\python.exe src\infer.py --checkpoint-prefix convnext_tiny --checkpoint-suffix best --tta-scales 256,288 --output result.txt
```

生成当前 V2 融合提交文件：

```powershell
.\.venv\Scripts\python.exe src\ensemble_infer.py --member convnext_tiny:convnext_tiny:0.6 --member efficientnet_v2_s:efficientnet_v2_s:0.4 --tta-scales 256,288 --output result.txt
.\.venv\Scripts\python.exe src\validate_submission.py result.txt
```

## 发布说明

- 默认不提交 `data/train/`、`data/test/`、模型权重、日志和 `result*.txt`
- `outputs/` 仅保留目录占位
- `temp/` 保持不动，作为历史实现备份

## 文档

- V1 方案：[docs/项目实施方案-V1.md](docs/项目实施方案-V1.md)
- V2 方案：[docs/项目实施方案-V2.md](docs/项目实施方案-V2.md)
- 发布检查：[docs/GitHub发布检查.md](docs/GitHub发布检查.md)
