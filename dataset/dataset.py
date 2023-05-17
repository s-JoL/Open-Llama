"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2023-04-24 20:05:21
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2023-05-06 23:30:37
FilePath: /Open-Llama/dataset/dataset.py
Description: 

Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
"""
import math
import torch
import random
from glob import glob
from datasets import load_dataset


random.seed(42)


def pretrain_transform(batch):
    # wudao preprocess
    if "title" in batch and "content" in batch:
        assert len(batch["title"]) == 1
        batch["text"] = [batch["title"][0] + "\n" + batch["content"][0]]
    elif "text" in batch:
        pass
    else:
        raise Exception("Unrecognized pretrain dataset format.")
    return batch


def instruct_transform(batch):
    # self instruct preprocess
    if "prompt" in batch and "completion" in batch:
        prompt = batch["prompt"][0]
        completion = batch["completion"][0]
        if prompt.endswith("Output:"):
            prompt = prompt[:-7]
        text = "user:{}\nsystem:{}".format(prompt.strip(), completion.strip())
        texts = [text]
    # belle preprocess
    elif "instruction" in batch and "output" in batch:
        prompt = batch["instruction"][0].replace("\\n", "")
        prompt = prompt.strip("")

        completion = batch["output"][0].replace("\\n", "")
        completion = completion.strip("")
        # multi turn chat
        if "Human:" in prompt:
            texts = []
            chats = prompt + completion
            chats = chats.split("Human:")
            for chat in chats:
                if chat.strip() == "":
                    continue
                res = chat.split("Assistant:")
                if len(res) != 2:
                    continue
                prompt, completion = res
                prompt = prompt.strip()
                completion = completion.strip()
                chat = "user:{}\nsystem:{}".format(prompt, completion)
                texts.append(chat)
            texts = ["[multiturn_sep]".join(texts)]
        else:
            text = "user:{}\nsystem:{}".format(prompt, completion)
            texts = [text]
    # instruct code preprocess
    elif "instruction" in batch and "answer" in batch:
        prompt = batch["instruction"][0].replace("\\n", "")
        prompt = prompt.strip("")

        completion = batch["answer"][0].replace("\\n", "")
        completion = completion.strip("")
        text = "user:{}\nsystem:{}".format(prompt, completion)
        texts = [text]
    # share gpt preprocess
    elif "conversations" in batch:
        chats = batch["conversations"][0]
        if chats[0]["from"] != "human":
            chats = chats[1:]
        texts = []
        for i in range(len(chats) // 2):
            prompt = chats[2 * i]
            completion = chats[2 * i + 1]
            if not (prompt["from"] == "human" and completion["from"] == "gpt"):
                continue
            prompt = prompt["value"]
            prompt = prompt.strip()
            completion = completion["value"]
            completion = completion.strip()
            chat = "user:{}\nsystem:{}".format(prompt, completion)
            texts.append(chat)
        texts = ["[multiturn_sep]".join(texts)]
    # xP3 preprocess
    elif "inputs" in batch and "targets" in batch:
        inputs = batch["inputs"][0]
        targets = batch["targets"][0]
        text = "user:{}\nsystem:{}".format(inputs.strip(), targets.strip())
        texts = [text]
    # camel-ai preprocess
    elif "message_1" in batch and "message_2" in batch:
        inputs = batch["message_1"][0]
        targets = batch["message_2"][0]
        text = "user:{}\nsystem:{}".format(inputs.strip(), targets.strip())
        texts = [text]
    # grade-school-math-instructions preprocess
    elif "INSTRUCTION" in batch and "RESPONSE" in batch:
        inputs = batch["INSTRUCTION"][0]
        targets = batch["RESPONSE"][0]
        text = "user:{}\nsystem:{}".format(inputs.strip(), targets.strip())
        texts = [text]
    else:
        raise Exception("Unrecognized instruct dataset format.")
    return {"text": texts}


def split_multiturn(batch):
    return {"text": batch["text"][0].split("[multiturn_sep]")}


def sample_sequence_gen(seq_length, eos_token_id):
    def sample_sequence(line):
        doc_length = line["input_ids"].shape[0]
        if doc_length <= seq_length:
            start = 0
        else:
            if random.random() < 1 / 4:
                start = 0
            else:
                start = random.randint(0, doc_length - seq_length)
        input_ids = line["input_ids"][start : start + seq_length]
        if input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        return {"input_ids": input_ids}

    return sample_sequence


def split_sequence_gen(seq_length):
    def split_sequence(batch):
        input_ids = batch["input_ids"][0]
        out = []
        while len(input_ids) >= (1 + len(out)) * seq_length:
            out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
        return {"input_ids": out}

    return split_sequence


def concat_multiple_sequence_gen(seq_length, pad_token_id):
    def concat_multiple_sequence(batch):
        concat_input_ids = torch.cat(batch["input_ids"], dim=0)
        length = concat_input_ids.shape[0]
        chunks = math.ceil(length / seq_length)
        pad_length = chunks * seq_length - length
        pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_token_id
        concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)
        input_ids = torch.chunk(concat_input_ids, chunks)
        return {"input_ids": input_ids}

    return concat_multiple_sequence


def get_labels_gen(pad_token_id):
    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        return {"labels": labels}

    return get_labels


def construct_dataset(
    dataset_config, tokenizer, return_raw_text=False, world_size=None
):
    all_data_files = []
    for name, pattern in dataset_config["data"].items():
        data_files = glob(pattern)
        assert len(data_files) > 0
        all_data_files.extend(data_files)
    random.shuffle(all_data_files)
    # 当shard可以被world_size整除时 split_dataset_by_node 会直接按shard进行划分，否则会读所有数据然后跳过一部分，可能会慢一点
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.distributed.split_dataset_by_node
    if world_size is not None:
        num_shards = len(all_data_files)
        all_data_files = all_data_files[: num_shards // world_size * world_size]
    dataset = load_dataset(
        "json", data_files=all_data_files, split="train", streaming=True
    )
    # shuffle
    dataset = dataset.shuffle(seed=42)
    # 文本预处理转换为统一格式
    if dataset_config["mode"] == "pretrain":
        dataset = dataset.map(pretrain_transform, batched=True, batch_size=1)
    elif dataset_config["mode"] == "instruct":
        dataset = dataset.map(instruct_transform, batched=True, batch_size=1)
        dataset = dataset.select_columns("text")
        dataset = dataset.map(split_multiturn, batched=True, batch_size=1)
    else:
        raise Exception("Dataset mode: {} not found.".format(dataset_config["mode"]))

    full_dataset = dataset

    # to visualize
    if return_raw_text:
        return full_dataset

    seq_length = dataset_config["seq_length"]
    pad_to_max = dataset_config.get("pad_to_max", True)
    sequence_sample_mode = dataset_config.get("sequence_sample_mode", "truncation")
    truncation = sequence_sample_mode == "truncation"
    concat_multiple_sequence = dataset_config.get("concat_multiple_sequence", False)
    # tokenize
    if pad_to_max:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                padding="max_length",
                max_length=seq_length,
                truncation=truncation,
            )
        )
    else:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                truncation=truncation,
            )
        )

    # format
    full_dataset = full_dataset.map(lambda x: {"input_ids": x["input_ids"][0]})
    full_dataset = full_dataset.select_columns("input_ids")

    # sequence_sample
    if sequence_sample_mode == "truncation":
        pass
    elif sequence_sample_mode == "none":
        pass
    elif sequence_sample_mode == "sample":
        assert pad_to_max or concat_multiple_sequence
        full_dataset = full_dataset.map(
            sample_sequence_gen(seq_length, tokenizer.eos_token_id)
        )
    elif sequence_sample_mode == "split":
        assert not concat_multiple_sequence
        full_dataset = full_dataset.map(
            split_sequence_gen(seq_length), batched=True, batch_size=1
        )
    else:
        raise Exception(
            "Unknown sequence_sample mode: {}.".format(sequence_sample_mode)
        )

    # concat multiple sequence
    if concat_multiple_sequence:
        num_sequences = dataset_config["num_sequences"]
        full_dataset = full_dataset.map(
            concat_multiple_sequence_gen(seq_length, tokenizer.pad_token_id),
            batched=True,
            batch_size=num_sequences,
            drop_last_batch=True,
        )

    # add label
    full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_token_id))

    # shuffle
    full_dataset = full_dataset.shuffle(seed=42)
    return full_dataset


if __name__ == "__main__":
    import time
    from unicodedata import normalize
    from torch.utils.data import DataLoader
    from transformers import LlamaTokenizer

    data_config = {
        "mode": "pretrain",
        "data": {"mixed": "data/pretrain_data/part-*.jsonl.zst"},
        "pad_to_max": False,
        "sequence_sample_mode": "sample",
        "concat_multiple_sequence": True,
        "num_sequences": 10,
        "seq_length": 2048,
    }
    tokenizer = LlamaTokenizer(
        "configs/tokenizer_models/llama_tokenizer_extended.model",
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
        )["input_ids"][0]
        decode_text = tokenizer.decode(input_ids, skip_special_tokens=True)
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
