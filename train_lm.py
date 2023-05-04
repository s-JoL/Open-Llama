"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-04-12 19:12:42
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-05-04 09:19:15
FilePath: /Open-Llama/train_lm.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import yaml
import torch
import logging
from absl import app
from absl import flags
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
from transformers import OpenLlamaForCausalLM, OpenLlamaConfig, LlamaTokenizer

from dataset.dataset import construct_dataset
from solver.trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Training config path")


def main(argv):
    with open(FLAGS.config, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get(
            "gradient_accumulation_steps", 1
        )
    )
    tokenizer = LlamaTokenizer(
        config["data"]["tokenizer_model_path"],
        pad_token="<pad>",
        add_bos_token=False,
        add_eos_token=True,
    )
    data_config = config["data"]
    if data_config.get("split_by_shard", False):
        train_dataset = construct_dataset(
            data_config, tokenizer, world_size=accelerator.num_processes
        )
    else:
        train_dataset = construct_dataset(data_config, tokenizer)
    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        num_workers=config["train"]["train_num_workers"],
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )
    # smaller initializer_range make training more stable
    # add stabel embedding to token embedding
    raw_model = OpenLlamaForCausalLM(
        OpenLlamaConfig(
            vocab_size=tokenizer.vocab_size,
            initializer_range=config["model"]["initializer_range"],
            pad_token_id=tokenizer.pad_token_id,
            rms_norm_eps=1e-5,
            hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
            attention_dropout_prob=config["model"]["attention_dropout_prob"],
            use_stable_embedding=config["model"]["use_stable_embedding"],
            shared_input_output_embedding=config["model"][
                "shared_input_output_embedding"
            ],
        )
    )
    if config["train"]["ckpt"] is not None:
        ckpt = torch.load(config["train"]["ckpt"], map_location="cpu")
        if "module" in ckpt:
            ckpt = ckpt["module"]
        raw_model.load_state_dict(ckpt)
        logging.warn("Loaded ckpt from: {}".format(config["train"]["ckpt"]))
    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
