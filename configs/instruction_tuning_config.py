"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-30 21:38:07
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-06 03:37:23
FilePath: /Open-Llama/configs/instruction_tuning_config.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
max_length = 1024
train_batch_size = 2
num_training_steps = 40000
num_warmup_steps = 100
initializer_range = 1e-2
lr = 2e-4
weight_decay = 1e-1
tokenizer_model_path = "configs/10w_vocab_wudao5_pile10.model"
patterns = ["data/instruction_data/part-*.jsonl.zst"]
# global step
log_interval = 50
eval_interval = 500
save_interval = 1000
work_dir = "data/saved_ckpt/"
ckpt_path = "data/saved_ckpt/83200.pt"
