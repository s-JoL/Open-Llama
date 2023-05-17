"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2023-04-12 19:12:42
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2023-05-17 22:20:32
FilePath: /Open-Llama/train_lm.py
Description: 

Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
"""
import yaml
import math
import logging
from absl import app
from absl import flags
from accelerate import Accelerator
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from datasets.distributed import split_dataset_by_node
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer

from dataset.dataset import construct_dataset
from solver.trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config", None, "Training config path")
flags.DEFINE_string(
    "model_config", "configs/model_configs/7B.json", "Model config path"
)


def main(argv):
    with open(FLAGS.train_config, "r", encoding="utf-8") as fp:
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
    model_config = AutoConfig.from_pretrained(FLAGS.model_config)
    # Make the vocab size divisible by 16
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#how-to-choose-which-zero-stage-and-offloads-to-use-for-best-performance
    # https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
    # vocab_size = math.ceil(tokenizer.vocab_size / 16) * 16
    # logging.warning(
    #     "Round vocab_size from {} to {}.".format(tokenizer.vocab_size, vocab_size)
    # )
    vocab_size = tokenizer.vocab_size
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = tokenizer.pad_token_id
    # 使用AutoModel可以在Deepspeed.zero.Init()下正确的生效，而直接使用如OpenLlamaModel不能正确生效，导致浪费大量内存空间
    # https://github.com/huggingface/accelerate/pull/932
    if config["train"]["ckpt"] is not None:
        raw_model = AutoModelForCausalLM.from_pretrained(
            config["train"]["ckpt"], config=model_config
        )
        logging.warning("Loaded ckpt from: {}".format(config["train"]["ckpt"]))
    else:
        raw_model = AutoModelForCausalLM.from_config(model_config)
    # lora
    if config["train"].get("use_lora", False):
        # gradient ckpt bug, https://github.com/huggingface/transformers/issues/23170
        if hasattr(raw_model, "enable_input_require_grads"):
            raw_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            raw_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        raw_model = get_peft_model(raw_model, peft_config)
        raw_model.print_trainable_parameters()
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()
    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
