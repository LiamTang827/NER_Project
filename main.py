import os
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


def tokenize_and_align_labels(batch, tokenizer, label_all_tokens=False, max_length=128):
    tokenized_inputs = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    labels = []
    for i, ner_tags in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(ner_tags[word_idx])
            else:
                label_ids.append(ner_tags[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    print("Loading dataset...")
    ds = load_dataset("conll2003")

    label_list = ds["train"].features["ner_tags"].feature.names
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    print("Preparing tokenizer and small subsets...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)

    # Small subsets for quick demo — increase or remove select() for full training
    train_small = ds["train"].select(range(min(400, len(ds["train"]))))
    val_small = ds["validation"].select(range(min(200, len(ds["validation"]))))
    test_small = ds["test"].select(range(min(200, len(ds["test"]))))

    tokenized_train = train_small.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=train_small.column_names)
    tokenized_val = val_small.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=val_small.column_names)
    tokenized_test = test_small.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=test_small.column_names)

    print("Building model...")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    use_fp16 = torch.cuda.is_available()
    print("CUDA available:" , torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir="outputs/main-demo",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=use_fp16,
        logging_steps=50,
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
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Start training (quick demo, 1 epoch)...")
    trainer.train()

    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized_test)
    print("Test results:", results)


if __name__ == "__main__":
    main()
