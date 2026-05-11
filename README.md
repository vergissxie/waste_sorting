# Waste Sorting

基于图像分类的垃圾分类模型项目。当前数据集按类别编号组织在 `data/train/` 下，测试图片位于 `data/test/`。当前稳定实现已经整理到 `src/`，`temp/` 仅保留为历史参考脚本。

## 版本说明

- **V1**：早期方案与历史实验记录，主要用于验证整体方向。
- **V2**：当前正式方案，代码入口位于 `src/`，采用 ConvNeXt Tiny + ImageNet 预训练 + 5-fold + Weighted CrossEntropy + fold ensemble + hflip TTA。

## v1 结果

- 方案：ConvNeXt Tiny ImageNet 预训练，5-fold 训练，推理阶段使用 fold ensemble + 水平翻转 TTA。
- 本地 5-fold 平均验证准确率：`90.46%`
- 本地 5-fold 平均 macro F1：`89.87%`
- 平台提交分数：`91.75`

## v2 当前实现

- 训练入口：`src/train.py`
- 推理入口：`src/infer.py`
- 提交校验：`src/validate_submission.py`
- 数据处理：`src/dataset.py`
- 模型定义：`src/model.py`
- 默认配置：`src/config.py`

当前 `src/config.py` 默认配置：

- 模型：`convnext_tiny`
- 类别数：`40`
- 输入尺寸：`256`
- 折数：`5`
- batch size：`16`
- epoch：`25`
- 学习率：`3e-4`
- weight decay：`1e-4`
- warmup：`3`
- Weighted CE：开启
- AMP：开启

## 项目结构

```text
waste_sorting/
  data/                  # 本地数据集，不建议直接提交到 GitHub
  docs/                  # 项目方案、结构说明、发布检查
  outputs/               # 模型权重、日志、预测结果
  scripts/               # 数据准备、清理、统计等辅助脚本
  src/                   # 当前正式源码目录
    config.py
    dataset.py
    model.py
    train.py
    infer.py
    validate_submission.py
  temp/                  # 历史/参考脚本，默认不再作为当前训练入口
  requirements.txt       # Python 依赖
```

详细说明见 [docs/项目结构说明.md](docs/项目结构说明.md)。

## 本地环境

建议使用 Python 3.9 或 3.10，并在虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要使用 NVIDIA GPU，建议按本机 CUDA 版本从 PyTorch 官方安装对应的 `torch` 和 `torchvision` 版本。

## 数据约定

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

当前训练集包含 40 个类别目录，测试集约 400 张图片。`data/train/` 和 `data/test/` 默认被 `.gitignore` 忽略，避免 GitHub 仓库体积过大。

## 后续实施

V1 方案见 [docs/项目实施方案-V1.md](docs/项目实施方案-V1.md)，V2 方案见 [docs/项目实施方案-V2.md](docs/项目实施方案-V2.md)。发布前检查见 [docs/GitHub发布检查.md](docs/GitHub发布检查.md)。

## 训练与推理

数据放好后，运行完整 5-fold 训练：

```powershell
.\.venv\Scripts\python.exe src\train.py
```

生成提交文件：

```powershell
.\.venv\Scripts\python.exe src\infer.py --output result.txt
.\.venv\Scripts\python.exe src\validate_submission.py result.txt
```
