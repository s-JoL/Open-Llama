"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-04-08 22:44:44
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-08 23:15:57
FilePath: /Open-Llama/speed_test/accelerate/run.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import time
import torch
from deepspeed.ops.adam import FusedAdam
from accelerate import Accelerator, DistributedType
from transformers import LlamaForCausalLM, LlamaConfig

batch_size = 32
seq_length = 2048
vocab_size = 32000
total_step = 2
use_activation_ckpt = True

class FakeSet(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return torch.randint(0, vocab_size, (seq_length, ))
    
    def __len__(self):
        return 1000000000

accelerator = Accelerator()
raw_model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=vocab_size,
    )
)
if use_activation_ckpt:
    raw_model.gradient_checkpointing_enable()
optimizer = FusedAdam(raw_model.parameters(), lr=1e-5)

train_loader = torch.utils.data.DataLoader(FakeSet(), batch_size=batch_size)
if accelerator.distributed_type == DistributedType.FSDP:
    accelerator.print('FSDP')
    model = accelerator.prepare(raw_model)
    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)
else:
    model, optimizer, train_loader = accelerator.prepare(raw_model, optimizer, train_loader)

def train(model, optimizer, train_loader):
    start_time = time.time()
    for i,  batch in enumerate(train_loader):
        if i == total_step:
            break
        optimizer.zero_grad()
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        accelerator.backward(loss)
        optimizer.step()
    end_time = time.time()
    return end_time - start_time

accelerator.print('total time: {}'.format(train(model, optimizer, train_loader)))
