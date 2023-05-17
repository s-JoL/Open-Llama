"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2023-04-11 20:07:35
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2023-04-11 21:56:07
FilePath: /Open-Llama/speed_test/lightning/run.py
Description: 

Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
"""
import time
import torch
import lightning.pytorch as pl
from deepspeed.ops.adam import FusedAdam
from transformers import LlamaForCausalLM, LlamaConfig
from lightning.pytorch.strategies import DeepSpeedStrategy


batch_size = 2
seq_length = 2048
vocab_size = 32000
total_step = 100
use_activation_ckpt = False


class FakeSet(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return torch.randint(0, vocab_size, (seq_length,))

    def __len__(self):
        return 1000000000


class SpeedTest(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=vocab_size,
            )
        )
        if use_activation_ckpt:
            self.model.gradient_checkpointing_enable()
        self.start_time = None

    def training_step(self, batch, batch_idx):
        out = self.model(batch, labels=batch)
        loss = out.loss
        if self.start_time is None:
            print("start")
            self.start_time = time.time()
        return loss

    def configure_optimizers(self):
        optimizer = FusedAdam(self.trainer.model.parameters(), lr=1e-5)
        return optimizer


model = SpeedTest()
train_loader = torch.utils.data.DataLoader(FakeSet(), batch_size=batch_size)

strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=False,
    offload_parameters=False,
    process_group_backend="nccl",
)
trainer = pl.Trainer(
    limit_train_batches=total_step,
    max_epochs=1,
    devices=8,
    accelerator="gpu",
    strategy=strategy,
    precision=16,
    enable_checkpointing=False,
)


def train(model, train_loader):
    start_time = time.time()
    trainer.fit(model=model, train_dataloaders=train_loader)
    end_time = time.time()
    return end_time - model.start_time


print("total time: {}".format(train(model, train_loader)))
