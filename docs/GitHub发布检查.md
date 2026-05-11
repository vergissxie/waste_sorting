# GitHub 发布检查

## 发布前检查

- 确认 `.gitignore` 已忽略数据图片、模型权重、日志和虚拟环境。
- 确认 `README.md` 能说明项目用途、安装方式、数据放置方式和运行入口。
- 确认 `requirements.txt` 可以安装基础依赖。
- 确认 `src/` 中没有硬编码的个人绝对路径。
- 确认没有提交 `.claude/`、`.venv/`、`runs/`、`outputs/` 里的运行产物。
- 确认数据集授权允许公开发布；不确定时不要上传原始图片。

## 建议提交内容

```text
README.md
requirements.txt
.gitignore
docs/
scripts/
src/
data/garbage_dict.json
data/testpath.txt
```

## 不建议提交内容

```text
data/train/
data/test/
outputs/
runs/
*.pth
*.pth.tar
*.pt
.venv/
.claude/
```

## 后续可选

- 使用 Git LFS 管理模型权重或小规模示例数据。
- 在 GitHub Release 中上传训练好的权重文件。
- 在 README 中增加数据集下载链接和权重下载链接。

