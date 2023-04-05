"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 21:02:00
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-06 03:33:27
FilePath: /Open-Llama/dataset/instruction_dataset.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import math


def preprocess_self_instruction_gen(tokenizer, segment_max_length=1024):
    def preprocess_self_instruction(line):
        """
        The format of the data is roughly as follows.
        {'prompt': 'Explain the origin of life on earth. Output:', 'completion': 'Life on Earth is believed to have'}
        Split the data based on the tokenized length according to the maximum length.
        """
        prompt = line["prompt"]
        if prompt.endswith("Output:"):
            prompt = prompt[:-7]
        total = "user:{}\nsystem:{}".format(prompt.strip(), line["completion"].strip())
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_self_instruction


def preprocess_belle_gen(tokenizer, segment_max_length=1024):
    def preprocess_belle(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        prompt = line["instruction"].replace("\\n", "")
        prompt = prompt.strip("")

        completion = line["output"].replace("\\n", "")
        completion = completion.strip("")
        total = "user:{}\nsystem:{}".format(prompt, completion)
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_belle


def preprocess_belle_multiturn_chat_gen(tokenizer, segment_max_length=1024):
    def preprocess_belle_multiturn_chat(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        prompt = line["instruction"].replace("\\n", "")
        prompt = prompt.strip("")

        completion = line["output"].replace("\\n", "")
        completion = completion.strip("")
        chats = prompt + completion
        chats = chats.split("Human:")
        input_ids = []
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
            out = tokenizer(chat)
            input_ids.extend(out["input_ids"])
        if len(input_ids) == 0:
            return None
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_belle_multiturn_chat


def preprocess_sharegpt_gen(tokenizer, segment_max_length=1024):
    def preprocess_sharegpt(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        chats = line["conversations"]
        if chats[0]["from"] != "human":
            chats = chats[1:]
        input_ids = []
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
            out = tokenizer(chat)
            input_ids.extend(out["input_ids"])
        if input_ids == []:
            return None
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_sharegpt


def preprocess_instruct_code_gen(tokenizer, segment_max_length=1024):
    def preprocess_instruct_code(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        prompt = line["instruction"].replace("\\n", "")
        prompt = prompt.strip("")

        completion = line["answer"].replace("\\n", "")
        completion = completion.strip("")
        total = "user:{}\nsystem:{}".format(prompt, completion)
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_instruct_code


if __name__ == "__main__":
    import sentencepiece as spm

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, DataIter

    sp_model = spm.SentencePieceProcessor(
        model_file="configs/10w_vocab_wudao5_pile10.model"
    )
    tokenizer = Tokenizer(sp_model)
    patterns = ["data/instruction_data/part-belle_multiturn_chat_0.8M-*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        "self_instruct": preprocess_self_instruction_gen(tokenizer),
        "belle_1M": preprocess_belle_gen(tokenizer),
        "belle_0.5M": preprocess_belle_gen(tokenizer),
        "belle_school_math_0.25M": preprocess_belle_gen(tokenizer),
        "belle_multiturn_chat_0.8M": preprocess_belle_multiturn_chat_gen(tokenizer),
        "instruct_to_code": preprocess_instruct_code_gen(tokenizer),
        "sharegpt_90K": preprocess_sharegpt_gen(tokenizer),
    }
    data_set = DataIter(
        paths, transform_dict=transform_dict, concat_docs=True, max_length=1024
    )
    for i, sample in enumerate(data_set):
        print(sp_model.decode(sample))
        if i == 1:
            break
