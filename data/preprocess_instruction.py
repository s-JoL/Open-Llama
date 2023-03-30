"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 20:52:10
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-30 20:52:12
FilePath: /Open-Llama/data/preprocess_instruction.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import json
import zstandard as zstd
from datasets import load_dataset

dataset = load_dataset("yizhongw/self_instruct")
write_path = "data/instruction_data/part-self_instruct-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in dataset["train"]:
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()

dataset = load_dataset("BelleGroup/generated_train_0.5M_CN")
write_path = "data/instruction_data/part-belle_0.5M-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in dataset["train"]:
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()

dataset = load_dataset("BelleGroup/generated_train_1M_CN")
write_path = "data/instruction_data/part-belle_1M-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in dataset["train"]:
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
