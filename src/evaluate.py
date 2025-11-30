import argparse
import json
import os
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report
from datasets import load_metric
from src.data import load_datasets, tokenize_and_align_labels, get_label_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--output_dir", default=None, help="Directory to save evaluation results. Defaults to model_dir.")
    return p.parse_args()


# 🔧 递归转换 numpy.int64 / numpy.float64 → Python 类型
def convert_numpy(o):
    if isinstance(o, dict):
        return {k: convert_numpy(v) for k, v in o.items()}
    if isinstance(o, list):
        return [convert_numpy(i) for i in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    args = parse_args()

    # 加载数据
    ds = load_datasets("conll2003")
    model_path = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if args.model_name_or_path is None else args.model_name_or_path,
        use_fast=True
    )
    tokenized_ds, label_list = tokenize_and_align_labels(ds, tokenizer)

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    metric = load_metric("seqeval")
    id2label = {i: l for i, l in enumerate(label_list)}

    preds = []
    labs = []

    print("🔍 Running evaluation...")

    # 手动推理（不会卡）
    for batch in tokenized_ds["test"]:
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits.argmax(-1).squeeze().tolist()
        gold = batch["labels"]

        sequence_pred = []
        sequence_gold = []

        for p_i, g_i in zip(logits, gold):
            if g_i != -100:
                sequence_pred.append(id2label[p_i])
                sequence_gold.append(id2label[g_i])

        preds.append(sequence_pred)
        labs.append(sequence_gold)

    # 评价指标
    results = metric.compute(predictions=preds, references=labs)
    print("\n===== Evaluation Results =====")
    print(results)

    # 分类报告
    cls_report = classification_report(labs, preds)
    print("\n===== Classification Report =====")
    print(cls_report)

    # 输出路径
    output_dir = args.output_dir if args.output_dir else args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # 🔧 转换 numpy → Python 类型
    clean_results = convert_numpy(results)

    # 保存 JSON
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    # 保存 Markdown
    report_path = os.path.join(output_dir, "eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# NER Evaluation Report\n\n")
        f.write(f"**Model**: {model_path}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Precision | {clean_results.get('overall_precision', 0):.4f} |\n")
        f.write(f"| Recall | {clean_results.get('overall_recall', 0):.4f} |\n")
        f.write(f"| F1 | {clean_results.get('overall_f1', 0):.4f} |\n")
        f.write(f"| Accuracy | {clean_results.get('overall_accuracy', 0):.4f} |\n")
        f.write(f"\n## Classification Report\n\n")
        f.write(f"```\n{cls_report}\n```\n")

    print(f"\n✅ Results saved to {results_path}")
    print(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
