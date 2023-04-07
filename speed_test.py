# import time
# import torch
# from colossalai.nn.optimizer import HybridAdam
# from deepspeed.ops.adam import FusedAdam
# from transformers import LlamaForCausalLM, LlamaConfig
# import lightning.pytorch as pl

# # define the LightningModule
# class LitAutoEncoder(pl.LightningModule):
#     def __init__(self):
#         super().__init__()

#     def training_step(self, inputs, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
#         # print(inputs.shape)
#         out = self.model(input_ids=inputs, labels=inputs)
#         loss = out.loss
#         return loss

#     def configure_optimizers(self):
#         optimizer = HybridAdam(self.parameters(), lr=1e-5)
#         return optimizer

#     def configure_sharded_model(self):
#         self.model = LlamaForCausalLM(
#             LlamaConfig(
#                 vocab_size=32000,
#                 initializer_range=0.001,
#                 pad_token_id=0,
#                 rms_norm_eps=1e-5,
#                 hidden_dropout_prob=0.1,
#                 attention_dropout_prob=0.1,
#                 use_stable_embedding=False,
#                 shared_input_output_embedding=False,
#             )
#         )


# # init the autoencoder
# autoencoder = LitAutoEncoder()
# trainer = pl.Trainer(limit_train_batches=500, max_epochs=1, accelerator='gpu', devices=8, strategy="colossalai", precision=16)
# class FakeSet(torch.utils.data.Dataset):
#     def __getitem__(self, idx):
#         return torch.randint(0, 32000, (2048, ))

#     def __len__(self):
#         return 10000
# train_loader = torch.utils.data.DataLoader(FakeSet(), batch_size=1)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# import time
# import torch
# from accelerate import Accelerator
# from deepspeed.ops.adam import FusedAdam
# from transformers import LlamaForCausalLM, LlamaConfig


# accelerator = Accelerator()
# raw_model = LlamaForCausalLM(
#     LlamaConfig(
#         vocab_size=32000,
#         initializer_range=0.001,
#         pad_token_id=0,
#         rms_norm_eps=1e-5,
#         hidden_dropout_prob=0.1,
#         attention_dropout_prob=0.1,
#         use_stable_embedding=False,
#         shared_input_output_embedding=False,
#     )
# )
# optimizer = FusedAdam(raw_model.parameters(), lr=1e-5)

# import random
# import sentencepiece as spm
# from dataset.tokenizer import Tokenizer
# from dataset.data_iter import create_shard_kwargs, DataIter
# from torch.utils.data import DataLoader

# max_length = 2048
# tokenizer_model_path = 'configs/10w_vocab_wudao5_pile10.model'
# sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
# tokenizer = Tokenizer(sp_model)

# paths = create_shard_kwargs(['1*'])
# random.shuffle(paths)
# data_set = DataIter(
#     paths
# )
# train_loader = DataLoader(
#     data_set,
#     batch_size=1
# )

# model, optimizer, train_loader = accelerator.prepare(raw_model, optimizer, train_loader)
# inputs = torch.randint(0, 32000, (1, 2048), device=accelerator.device)


# for i in range(10):
#     optimizer.zero_grad()
#     out = model(input_ids=inputs, labels=inputs)
#     loss = out.loss
#     accelerator.backward(loss)
#     optimizer.step()
# start_time = time.time()
# for i in range(500):
#     optimizer.zero_grad()
#     out = model(input_ids=inputs, labels=inputs)
#     loss = out.loss
#     accelerator.backward(loss)
#     optimizer.step()
# end_time = time.time()
# accelerator.print(end_time - start_time)