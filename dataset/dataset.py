"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-04-24 20:05:21
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-24 20:05:59
FilePath: /Open-Llama/dataset/dataset.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import random
from glob import glob
from datasets import load_dataset, interleave_datasets


def pretrain_transform(line):
    if "title" in line and "text" not in line:
        line["text"] = line["title"] + "\n" + line["content"]
    return line


def sample_sequence_gen(seq_length, eos_token_id):
    def sample_sequence(line):
        doc_length = line["input_ids"].shape[1]
        if doc_length <= seq_length:
            start = 0
        else:
            if random.random() < 1 / 4:
                start = 0
            else:
                start = random.randint(0, doc_length - seq_length)
        input_ids = line["input_ids"][0, start : start + seq_length]
        if input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        return {"input_ids": input_ids}

    return sample_sequence


def concat_multiple_sequence_gen(seq_length):
    def concat_multiple_sequence(batch):
        concat_input_ids = torch.cat(batch["input_ids"], dim=0)
        input_ids = []
        while len(concat_input_ids) > (1 + len(input_ids)) * seq_length:
            input_ids.append(
                concat_input_ids[
                    len(input_ids) * seq_length : (1 + len(input_ids)) * seq_length
                ]
            )
        out = {"input_ids": input_ids}
        return out

    return concat_multiple_sequence


def get_labels_gen(pad_token_id):
    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        return {"labels": labels}

    return get_labels


def construct_dataset(dataset_config, tokenizer, return_raw_text=False):
    datasets = []
    probabilities = []
    # 暂时只使用一个，使用多个时无法使用多进程读取导致耗时较长
    assert len(dataset_config["data"]) == 1
    for name, pattern in dataset_config["data"].items():
        data_files = glob(pattern)
        assert len(data_files) > 0
        dataset = load_dataset(
            "json", data_files=data_files, split="train", streaming=True
        )
        if dataset_config["mode"] == "pretrain":
            dataset = dataset.map(pretrain_transform)
        else:
            raise Exception(
                "Dataset mode: {} not found.".format(dataset_config["mode"])
            )
        datasets.append(dataset)
        probabilities.append(dataset.n_shards)
    probabilities_sum = sum(probabilities)
    probabilities = [p / probabilities_sum for p in probabilities]
    if len(datasets) > 1:
        full_dataset = interleave_datasets(
            datasets, probabilities=probabilities, seed=42
        )
    else:
        full_dataset = datasets[0]
    if return_raw_text:
        return full_dataset
    seq_length = dataset_config["seq_length"]
    if dataset_config.get("concat_multiple_sequence", False):
        num_sequences = dataset_config["num_sequences"]
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"], return_tensors="pt", return_attention_mask=False
            )
        )
        full_dataset = full_dataset.map(
            sample_sequence_gen(seq_length, tokenizer.eos_token_id)
        )
        full_dataset = full_dataset.select_columns("input_ids")
        full_dataset = full_dataset.map(
            concat_multiple_sequence_gen(seq_length),
            batched=True,
            batch_size=num_sequences,
        )
    else:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                padding="max_length",
                max_length=seq_length,
                truncation=True,
            )
        )
        full_dataset = full_dataset.map(lambda x: {"input_ids": x["input_ids"][0]})
        full_dataset = full_dataset.select_columns("input_ids")
    full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_token_id))
    return full_dataset


if __name__ == "__main__":
    import time
    from unicodedata import normalize
    from torch.utils.data import DataLoader
    from transformers import OpenLlamaTokenizer

    data_config = {
        "mode": "pretrain",
        "data": {"wudao": "data/pretrain_data/part-wudao*.jsonl.zst"},
        "concat_multiple_sequence": True,
        "num_sequences": 10,
        "seq_length": 2048,
    }
    tokenizer = OpenLlamaTokenizer(
        "configs/llama_tokenizer_extended.model",
        pad_token="<pad>",
        add_bos_token=False,
        add_eos_token=True,
    )
    pretrain_dataset = construct_dataset(data_config, tokenizer, True)
    start = time.time()
    for i, line in enumerate(pretrain_dataset):
        raw_text = line["text"]
        # raw_text = normalize("NFKC", raw_text)
        input_ids = tokenizer(
            line["text"], return_tensors="pt", return_attention_mask=False
        )["input_ids"][0, :-1]
        decode_text = tokenizer.decode(input_ids)
        if raw_text != decode_text and "▁" not in raw_text:
            print(raw_text, "\n", decode_text)
        if i == 3000:
            break
    print("all checked in {} seconds.".format(time.time() - start))
    pretrain_dataset = construct_dataset(data_config, tokenizer)
    print(pretrain_dataset.n_shards)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, num_workers=16)
    for batch in pretrain_loader:
        for k, v in batch.items():
            print(k, v.shape, "\n", v)
        break
