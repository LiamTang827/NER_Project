NER_Project demo

Goal
- Demo NER extractor: token-classification model on CoNLL-2003.

Setup (macOS, zsh)
1. Create & activate venv (if not already):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install pinned requirements:

```bash
pip install -r requirements.txt
```

Quick run (GPU recommended)
- Train (example):

```bash
python src/train.py --model_name_or_path bert-base-cased --output_dir outputs/bert --epochs 3 --per_device_train_batch_size 8
```

- Evaluate:

```bash
python src/evaluate.py --model_dir outputs/bert
```

Files
- `src/data.py` - dataset loading + tokenization + label alignment
- `src/train.py` - training script using `transformers.Trainer`
- `src/evaluate.py` - evaluation (seqeval)

Notes
- The demo uses the HuggingFace `conll2003` dataset by default. If you have local CoNLL files, modify `src/data.py::load_datasets` to point to them.
- For remote dataset scripts, if you trust them you can pass `trust_remote_code=True` to `load_dataset`.
Windows (PowerShell) GPU notes
- Create & activate venv (PowerShell):

```powershell
python -m venv .venv
# Activate (you may need to allow script execution once):
# Run PowerShell as Administrator and run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

- Install PyTorch with CUDA support (pick the CUDA version that matches your GPU/drivers). Example for CUDA 11.8:

```powershell
# CPU-only (simple):
pip install -r requirements.txt

# GPU (example CUDA 11.8) - install torch first using official wheels, then other deps:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

- Run training (PowerShell):

```powershell
python src/train.py --model_name_or_path bert-base-cased --output_dir outputs/bert --epochs 3 --per_device_train_batch_size 8
```

Notes:
- If you installed `torch` from the PyTorch CUDA wheels, `transformers`/`Trainer` will use CUDA automatically if available. The demo `train.py` auto-detects CUDA and enables `fp16` only when CUDA is available.
- If you prefer, create a `requirements-gpu.txt` that omits `torch` and install `torch` manually via the PyTorch installer command above.
