import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def load_datasets(dataset_name="conll2003", local_dir=None):
    """Load dataset. If local_dir provided, expects train.txt/valid.txt/test.txt in the dir.
    Otherwise loads from Hugging Face: `conll2003`.
    """
    if local_dir:
        # Not implemented: for demo we rely on HF dataset
        raise NotImplementedError("Local parser not implemented in this demo; use HF dataset or extend this function.")
    else:
        # Use HF built-in dataset. If remote code is used and trusted, add trust_remote_code=True
        ds = load_dataset(dataset_name)
        return ds


def get_label_list(dataset):
    # dataset is a DatasetDict
    # ner_tags is a Sequence(ClassLabel), get inner ClassLabel names
    label_feature = dataset["train"].features["ner_tags"].feature
    return label_feature.names


def tokenize_and_align_labels(dataset, tokenizer_name_or_obj, label_all_tokens=False, max_length=128):
    if isinstance(tokenizer_name_or_obj, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_obj, use_fast=True)
    else:
        tokenizer = tokenizer_name_or_obj

    label_list = get_label_list(dataset)

    def tokenize_batch(examples):
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
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

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized, label_list
