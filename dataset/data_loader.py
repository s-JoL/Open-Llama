"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 20:58:16
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-30 21:00:49
FilePath: /Open-Llama/dataset/data_loader.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import math
import torch


def pretrain_collate_fn_gen(tokenizer, segment_max_length=1024, padding="longest"):
    """
    Organize data into tensors by padding based on the preset maximum length.
    """
    pad_id = tokenizer.pad_id

    def pretrain_collate_fn(batch):
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

    return pretrain_collate_fn


class BySequenceLengthDataset(torch.utils.data.IterableDataset):
    """
    experimental
    """

    def __init__(
        self, generator, batch_size, accelerator=None, bucket_size=16, max_length=1024
    ):
        super().__init__()
        self.generator = generator
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.bucket_num = math.ceil(max_length / bucket_size)
        self.buckets = [[] for _ in range(self.bucket_num)]
        self.bucket_idx = None
        self.accelerator = accelerator
        if self.accelerator is not None:
            self.buckets_ele_num = torch.tensor(
                [0] * self.bucket_num, dtype=torch.int64, device=accelerator.device
            )
            self.buckets_indexes = torch.arange(
                self.bucket_num, device=accelerator.device
            )
        self.finished = False
        self.has_no_same_bucket = False
        self.rest = None

    def __iter__(self):
        if self.batch_size <= 1:
            return self.generator

        def bucket_iter():
            while True:
                if self.bucket_idx is not None:
                    sample = self.buckets[self.bucket_idx].pop()
                    if len(self.buckets[self.bucket_idx]) == 0:
                        self.bucket_idx = None
                    yield sample
                try:
                    sample = next(self.generator)
                except StopIteration:
                    break
                sample_len = len(sample) - 1
                bucket_idx = sample_len // self.bucket_size
                if len(self.buckets[bucket_idx]) == self.batch_size - 1:
                    self.bucket_idx = bucket_idx
                    yield sample
                else:
                    self.buckets[bucket_idx].append(sample)

        def parallel_bucket_iter():
            while True:
                if self.bucket_idx is not None:
                    sample = self.buckets[self.bucket_idx].pop()
                    self.buckets_ele_num[self.bucket_idx] -= 1
                    buckets_ele_num = self.accelerator.gather(self.buckets_ele_num)
                    buckets_ele_num = buckets_ele_num.reshape(
                        self.accelerator.num_processes, self.bucket_num
                    )
                    min_buckets_ele_num = buckets_ele_num.min(dim=0)[0]
                    if min_buckets_ele_num[self.bucket_idx] <= 0:
                        self.bucket_idx = None
                    yield sample
                else:
                    if self.finished:
                        if self.has_no_same_bucket:
                            if self.rest is None:
                                self.rest = []
                                for bucket in self.buckets:
                                    for i in bucket:
                                        self.rest.append(i)
                            elif len(self.rest) > 0:
                                yield self.rest.pop()
                            else:
                                raise StopIteration()
                        else:
                            buckets_ele_num = self.accelerator.gather(
                                self.buckets_ele_num
                            )
                            buckets_ele_num = buckets_ele_num.view(
                                self.accelerator.num_processes, self.bucket_num
                            )
                            min_buckets_ele_num = buckets_ele_num.min(dim=0)[0]
                            valid_bucket_idx = self.buckets_indexes[
                                min_buckets_ele_num >= self.batch_size
                            ]
                            if len(valid_bucket_idx) > 0:
                                self.bucket_idx = valid_bucket_idx[0].cpu().item()
                            else:
                                self.has_no_same_bucket = True
                    else:
                        try:
                            sample = next(self.generator)
                        except StopIteration:
                            self.finished = True
                            continue
                        sample_len = len(sample) - 1
                        bucket_idx = sample_len // self.bucket_size
                        self.buckets[bucket_idx].append(sample)
                        self.buckets_ele_num[bucket_idx] += 1
                        buckets_ele_num = self.accelerator.gather(
                            self.buckets_ele_num
                        ).cpu()
                        buckets_ele_num = buckets_ele_num.view(
                            self.accelerator.num_processes, self.bucket_num
                        )
                        min_buckets_ele_num = buckets_ele_num.min(dim=0)[0]
                        valid_bucket_idx = self.buckets_indexes[
                            min_buckets_ele_num >= self.batch_size
                        ]
                        if len(valid_bucket_idx) > 0:
                            self.bucket_idx = valid_bucket_idx[0].cpu().item()

        if self.accelerator:
            return parallel_bucket_iter()
        else:
            return bucket_iter()


if __name__ == "__main__":
    import sentencepiece as spm
    from datasets import IterableDataset
    from torch.utils.data import DataLoader

    from dataset.pretrain_dataset import preprocess_wudao_gen, preprocess_the_pile_gen

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, create_data_iter

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
    data_set = IterableDataset.from_generator(
        create_data_iter, gen_kwargs={"paths": paths, "transform_dict": transform_dict}
    )
    train_loader = DataLoader(
        data_set,
        batch_size=8,
        num_workers=4,
        collate_fn=pretrain_collate_fn_gen(tokenizer),
        drop_last=True,
    )
    for batch in train_loader:
        for k, v in batch.items():
            print(k, v.shape)
        break
