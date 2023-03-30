"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 21:02:00
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-30 21:02:06
FilePath: /Open-Llama/dataset/instruction_dataset.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""


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
        total = "user:{}<s>system:{}".format(prompt.strip(), line["completion"].strip())
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [input_ids]

    return preprocess_self_instruction


def preprocess_belle_gen(tokenizer, segment_max_length=1024):
    def preprocess_belle(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        prompt = line["input"].replace("\\n", "")
        prompt = prompt.strip("")

        completion = line["target"].replace("\\n", "")
        completion = completion.strip("")
        total = "user:{}<s>system:{}".format(prompt, completion)
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [input_ids]

    return preprocess_belle


if __name__ == "__main__":
    import sentencepiece as spm
    from datasets import IterableDataset

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, create_data_iter

    sp_model = spm.SentencePieceProcessor(
        model_file="configs/10w_vocab_wudao5_pile10.model"
    )
    tokenizer = Tokenizer(sp_model)
    patterns = ["data/instruction_data/part-belle_1M*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        "belle_1M": preprocess_belle_gen(tokenizer),
        "belle_0.5M": preprocess_belle_gen(tokenizer),
        "self_instruct": preprocess_self_instruction_gen(tokenizer),
    }
    data_set = IterableDataset.from_generator(
        create_data_iter, gen_kwargs={"paths": paths, "transform_dict": transform_dict}
    )
    for i, sample in enumerate(data_set):
        print(sample, sp_model.Decode(sample))
        if i == 20:
            break
