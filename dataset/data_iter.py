"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-17 19:32:20
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-05 22:36:45
FilePath: /Open-Llama/dataset/data_iter.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import json
from glob import glob
import zstandard as zstd
from torch.utils.data import IterableDataset


class DataIter(IterableDataset):
    """
    Currently, the allowed storage formats are jsonl.zst.
    Each line of the data is a dictionary, which can be parsed as JSON for subsequent processing after reading.
    Currently, only single worker is supported.
    """

    def __init__(
        self,
        paths_with_index,
        transform_dict=None,
        max_length=None,
        concat_docs=False,
        process_index=0,
        num_processes=1,
    ):
        super().__init__()
        self.paths_with_index = paths_with_index
        self.max_length = max_length
        self.transform_dict = transform_dict
        self.concat_docs = concat_docs
        self.process_index = process_index
        self.num_processes = num_processes
        if self.concat_docs:
            self.cache = []

    def __iter__(self):
        past = None
        for i, path in self.paths_with_index:
            # part-dataset_name-01.jsonl.zst
            dataset_name = path.split("-")[-2]
            # shard to multiple device
            if self.num_processes > 1 and i % self.num_processes != self.process_index:
                continue
            # Log the file name when encountering a new file.
            if past != dataset_name:
                print("Loading data from {}".format(path))
                past = path
            # Currently, the allowed storage formats are jsonl.zst.
            assert path.endswith("jsonl.zst")
            with zstd.open(path, "r", encoding="utf-8") as fp:
                for line in fp:
                    # If the length of the cache is greater than max_length.
                    if self.concat_docs and len(self.cache) >= self.max_length:
                        seq = self.cache[: self.max_length]
                        self.cache = self.cache[self.max_length :]
                        yield seq
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    line = json.loads(line)
                    line["dataset"] = dataset_name
                    # Transformation, including sample, tokenize, etc.
                    if self.transform_dict:
                        line = self.transform_dict[dataset_name](line)
                        if isinstance(line, str):
                            yield line
                        # must be list of list
                        elif isinstance(line, list) and isinstance(line[0], list):
                            for seq in line:
                                if self.concat_docs:
                                    # concat seq from multiple docs
                                    self.cache += seq
                                else:
                                    yield seq
                        else:
                            raise Exception(
                                "Unsupported type in Transformation: {}".format(
                                    self.transform_dict[dataset_name]
                                )
                            )
                    else:
                        yield line


def create_shard_kwargs(patterns, repeat=1):
    """
    Assign numbers to different shards of data to ensure that data is not duplicated
    when allocated to different nodes during distributed training.
    """
    all_path = []
    for p in patterns:
        all_path.extend(glob(p))
    all_path *= repeat
    return [(i, p) for i, p in enumerate(all_path)]


if __name__ == "__main__":
    patterns = ["data/pretrain_data/part-wudao*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {"wudao": lambda x: x["title"], "pile": lambda x: [x["text"]]}
    data_iter = DataIter(
        paths, transform_dict=transform_dict, max_length=16, concat_docs=True
    )
    for i, data in enumerate(data_iter):
        print(i, data)
        if i == 20:
            break
