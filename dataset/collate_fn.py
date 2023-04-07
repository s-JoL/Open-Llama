"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 20:58:16
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-05 22:11:03
FilePath: /Open-Llama/dataset/collate_fn.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch


def collate_fn_gen(tokenizer, segment_max_length=1024, padding="longest"):
    """
    Organize data into tensors by padding based on the preset maximum length.
    """
    pad_id = tokenizer.pad_id

    def collate_fn(batch):
        if padding == "longest":
            max_length = max([len(i) for i in batch])
        elif padding == "max_length":
            max_length = segment_max_length
        else:
            raise Exception("Invalid argumet for padding: {}".format(padding))
        input_ids = []
        for i in batch:
            input_len = len(i)
            input_ids.append(i + [pad_id] * (max_length - input_len))
        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        }
        return inputs

    return collate_fn


if __name__ == "__main__":
    import sentencepiece as spm
    from torch.utils.data import DataLoader

    from dataset.pretrain_dataset import preprocess_wudao_gen, preprocess_the_pile_gen

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, DataIter

    sp_model = spm.SentencePieceProcessor(
        model_file="configs/10w_vocab_wudao5_pile10.model"
    )
    tokenizer = Tokenizer(sp_model)
    patterns = ["data/pretrain_data/part-*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        "wudao": preprocess_wudao_gen(tokenizer),
        "pile": preprocess_the_pile_gen(tokenizer),
    }
    data_set = DataIter(paths, transform_dict=transform_dict)
    train_loader = DataLoader(
        data_set,
        batch_size=8,
        num_workers=4,
        collate_fn=collate_fn_gen(tokenizer),
        drop_last=True,
    )
    for batch in train_loader:
        for k, v in batch.items():
            print(k, v.shape)
        break
