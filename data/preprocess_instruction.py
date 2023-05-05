"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 20:52:10
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-05-04 08:32:04
FilePath: /Open-Llama/data/preprocess_instruction.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import json
from tqdm import tqdm
import zstandard as zstd
from datasets import load_dataset


root_dir = "data"

dataset = load_dataset("yizhongw/self_instruct")
write_path = root_dir + "/instruction_data/part-self_instruct-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "yizhongw/self_instruct preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("BelleGroup/train_0.5M_CN")
write_path = root_dir + "/instruction_data/part-belle_0.5M-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "BelleGroup/train_0.5M_CN preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("BelleGroup/train_1M_CN")
write_path = root_dir + "/instruction_data/part-belle_1M-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "BelleGroup/train_1M_CN preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("BelleGroup/train_2M_CN")
write_path = root_dir + "/instruction_data/part-belle_2M-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "BelleGroup/train_2M_CN preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("BelleGroup/school_math_0.25M")
write_path = root_dir + "/instruction_data/part-belle_school_math_0.25M-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "BelleGroup/school_math_0.25M preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("BelleGroup/multiturn_chat_0.8M")
write_path = root_dir + "/instruction_data/part-belle_multiturn_chat_0.8M-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "BelleGroup/multiturn_chat_0.8M preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("Graverman/Instruct-to-Code")
write_path = root_dir + "/instruction_data/part-instruct_to_code-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "Graverman/Instruct-to-Code preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

# dataset = load_dataset("bigscience/xP3mt", "en")
# write_path = root_dir + "/instruction_data/part-bigscience/xP3mt_en-{}.jsonl.zst"
# total_num = 0
# file_num = 1
# wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
# for line in tqdm(dataset["train"]):
#     line = json.dumps(line)
#     if total_num % 1024 == 0 and total_num > 0:
#         file_num += 1
#         wfp.close()
#         wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
#     wfp.write(line.encode("utf-8"))
#     wfp.write(b"\n")
#     total_num += 1
# wfp.close()
# print(
#     "bigscience/xP3mt_en preprocess done. Total line: {}, Total file: {}".format(
#         total_num, file_num
#     )
# )

dataset = load_dataset("bigscience/xP3mt", "code")
write_path = root_dir + "/instruction_data/part-xP3mt_code-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "bigscience/xP3mt_code preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

dataset = load_dataset("bigscience/xP3mt", "zh")
write_path = root_dir + "/instruction_data/part-xP3mt_zh-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for line in tqdm(dataset["train"]):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "bigscience/xP3mt_zh preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)

write_path = root_dir + "/instruction_data/part-sharegpt_90K-{}.jsonl.zst"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
with open("{}/sg_90k_part1_html_cleaned.json".format(root_dir), "r") as fp:
    data1 = json.load(fp)
with open("{}/sg_90k_part2_html_cleaned.json".format(root_dir), "r") as fp:
    data2 = json.load(fp)
data = data1 + data2
for line in tqdm(data):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "anon8231489123/ShareGPT_Vicuna_unfiltered preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)
