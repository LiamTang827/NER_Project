
# NER_Project 一键复现说明

## 目标
基于 HuggingFace Transformers，复现英文 CoNLL-2003 命名实体识别（NER）任务，支持 macOS/Linux/Windows（含外星人电脑）。

---

## 1. 环境准备

### macOS/Linux (zsh/bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
# 如遇权限问题，先以管理员运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
# CPU-only:
pip install -r requirements.txt
# GPU (如需 CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 2. 一键训练与评测

### 训练（默认3轮，BERT-base，自动检测CUDA）
```bash
python src/train.py --model_name_or_path bert-base-cased --output_dir outputs/bert --epochs 3 --per_device_train_batch_size 8
```

### 评测
```bash
python src/evaluate.py --model_dir outputs/bert
```

### 一键全流程（可直接复制粘贴）
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python src/train.py --model_name_or_path bert-base-cased --output_dir outputs/bert --epochs 3 --per_device_train_batch_size 8 && python src/evaluate.py --model_dir outputs/bert
```

---

## 3. 主要文件说明
- `src/data.py`：数据加载、分词、标签对齐
- `src/train.py`：训练主脚本（transformers.Trainer）
- `src/evaluate.py`：评测脚本（seqeval）
- `main.py`：极简一键 demo（可选）

---

## 4. 输出文件说明

训练和评测完成后，结果自动保存到 `outputs/bert/` 目录：

| 文件 | 说明 |
|------|------|
| `report.md` | 训练汇总报告（Markdown） |
| `train_results.json` | 训练指标（JSON） |
| `eval_results.json` | 验证集指标（JSON） |
| `test_results.json` | 测试集指标（JSON） |
| `eval_report.md` | 评测详细报告（含分类报告） |

---

## 5. 常见问题
- **依赖安装慢/失败**：可用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **PyTorch CUDA 版本不符**：请根据你的显卡和驱动选择合适的 CUDA wheel（见 https://pytorch.org/get-started/locally/）
- **seqeval 安装失败**：确保 requirements.txt 里是 `seqeval==1.2.2`，如有问题用镜像源。
- **本地 CoNLL 数据**：如需自定义数据，修改 `src/data.py::load_datasets`。

---

## 6. 版本控制建议
- `.venv/`、`outputs/`、`__pycache__/` 等已加入 `.gitignore`，不要提交。
- 只需提交代码、README、requirements.txt。

---

## 7. 参考命令
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python src/train.py --model_name_or_path bert-base-cased --output_dir outputs/bert --epochs 3 --per_device_train_batch_size 8
python src/evaluate.py --model_dir outputs/bert
```

---

## 8. 报告自动收集脚本
如果你在训练机器上已经有 `outputs/bert/`（包含 `report.md` / `eval_report.md` / `*results*.json`），可以使用仓库内的脚本一键收集并提交报告：

```bash
./scripts/publish_reports.sh outputs/bert main
```

脚本会把报告复制到 `reports/`，添加 `reports/README.md`，并尝试将变更提交到 `origin/main`（请确保本地配置了 git 并有推送权限）。

不要把模型文件提交到仓库 —— 脚本只会收集小文本/JSON 报告文件。
