# NER Project - Reports

This directory contains evaluation reports, metrics, and visualizations for the NER project.

> ⚠️ **Note:** Do NOT commit model weights or the entire `outputs/` directory. Large model files should be stored on the Hugging Face Hub or other artifact storage.

## Directory Structure

```
reports/
├── figures/           # Generated visualizations
│   ├── entity_distribution.png
│   ├── sentence_length_distribution.png
│   ├── f1_scores_by_entity.png
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── tsne_embeddings.png
│   ├── pca_embeddings.png
│   └── model_comparison.png
├── metrics/           # JSON result files
│   ├── all_results.json
│   ├── eval_results.json
│   ├── test_results.json
│   └── train_results.json
└── bert/              # Model-specific reports
    ├── config.json
    ├── eval_report.md
    └── report.md
```

## Results Summary

### BERT-base-cased (Test Set)

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| PER    | 0.9486    | 0.9591 | 0.9538   | 1,615   |
| ORG    | 0.8952    | 0.9055 | 0.9003   | 1,661   |
| LOC    | 0.9287    | 0.9298 | 0.9292   | 1,666   |
| MISC   | 0.7811    | 0.8134 | 0.7969   | 702     |
| **Overall** | **0.9056** | **0.9165** | **0.9111** | 5,644 |

### Model Comparison

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | - | - | - |
| BiLSTM-CRF | - | - | - |
| DistilBERT | - | - | - |
| **BERT-base-cased** | **0.9056** | **0.9165** | **0.9111** |
| RoBERTa-base | - | - | - |

*Note: Fill in results for other models as they become available.*

## Generating Reports

Run the Jupyter notebook to regenerate all figures:

```bash
cd notebooks
jupyter notebook NER_Project_Report.ipynb
```

Or run all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute NER_Project_Report.ipynb
```
