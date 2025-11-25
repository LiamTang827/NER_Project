import argparse
import os
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer,
)
from datasets import load_metric
from src.data import load_datasets, tokenize_and_align_labels, get_label_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--output_dir", default="outputs/ner-demo")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_datasets("conll2003")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenized_ds, label_list = tokenize_and_align_labels(ds, tokenizer, max_length=args.max_length)

    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        logging_steps=50,
        push_to_hub=False,
    )

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(-1)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(predictions, labels):
            seq_pred = []
            seq_lab = []
            for p_i, l_i in zip(pred, lab):
                if l_i != -100:
                    seq_pred.append(id2label[p_i])
                    seq_lab.append(id2label[l_i])
            true_predictions.append(seq_pred)
            true_labels.append(seq_lab)
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
