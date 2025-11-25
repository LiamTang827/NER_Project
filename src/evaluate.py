import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_metric
from src.data import load_datasets, tokenize_and_align_labels, get_label_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name_or_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_datasets("conll2003")
    model_path = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path if args.model_name_or_path is None else args.model_name_or_path, use_fast=True)
    tokenized_ds, label_list = tokenize_and_align_labels(ds, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(model_path)

    metric = load_metric("seqeval")

    id2label = {i: l for i, l in enumerate(label_list)}

    preds = []
    labs = []
    for batch in tokenized_ds["test"]:
        # single example prediction (small-scale demo)
        import torch
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

    results = metric.compute(predictions=preds, references=labs)
    print(results)


if __name__ == "__main__":
    main()
