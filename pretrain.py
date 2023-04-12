import yaml
import torch
import random
from absl import app
from absl import flags
import sentencepiece as spm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaConfig

from dataset.tokenizer import Tokenizer
from dataset.data_iter import create_shard_kwargs, DataIter
from dataset.collate_fn import collate_fn_gen
from dataset.pretrain_dataset import (
    preprocess_the_pile_gen,
    preprocess_wudao_gen,
)
from solver.trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Training config path")

def main(argv):
    accelerator = Accelerator()

    with open(FLAGS.config, 'r', encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    sp_model = spm.SentencePieceProcessor(model_file=config['data']['tokenizer_model_path'])
    tokenizer = Tokenizer(sp_model)

    paths = create_shard_kwargs(config['data']['patterns'])
    random.shuffle(paths)
    transform_dict = {
        "wudao": preprocess_wudao_gen(tokenizer, config['model']['max_length']),
        "pile": preprocess_the_pile_gen(tokenizer, config['model']['max_length']),
    }
    data_set = DataIter(
        paths,
        transform_dict=transform_dict,
        concat_docs=True,
        max_length=config['model']['max_length'],
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    train_loader = DataLoader(
        data_set,
        batch_size=config['train']['train_batch_size'],
        # If num_workers is greater than 1, duplicate data may occur.
        num_workers=0,
        collate_fn=collate_fn_gen(tokenizer, config['model']['max_length']),
        drop_last=True,
    )
    # smaller initializer_range make training more stable
    # add stabel embedding to token embedding
    raw_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            initializer_range=config['model']['initializer_range'],
            pad_token_id=tokenizer.pad_id,
            rms_norm_eps=1e-5,
            hidden_dropout_prob=config['model']['hidden_dropout_prob'],
            attention_dropout_prob=config['model']['attention_dropout_prob'],
            use_stable_embedding=config['model']['use_stable_embedding'],
            shared_input_output_embedding=config['model']['shared_input_output_embedding'],
        )
    )
    if config['train']['ckpt'] is not None:
        ckpt = torch.load(config['train']['ckpt'])
        raw_model.load_state_dict(ckpt)
    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)
    trainer.train()

if __name__ == '__main__':
    app.run(main)