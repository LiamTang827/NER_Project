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
