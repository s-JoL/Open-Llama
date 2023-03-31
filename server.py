"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-31 13:26:15
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-03-31 14:05:35
FilePath: /Open-Llama/server.py
Description: 

Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import gradio as gr
import sentencepiece as spm
from dataset.tokenizer import Tokenizer
from transformers import LlamaForCausalLM, LlamaConfig


sp_model = spm.SentencePieceProcessor(
    model_file="configs/10w_vocab_wudao5_pile10.model"
)
tokenizer = Tokenizer(sp_model)

raw_model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        initializer_range=0.01,
        pad_token_id=tokenizer.pad_id,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        use_stable_embedding=True,
        shared_input_output_embedding=True,
    )
)
ckpt = torch.load("data/saved_ckpt/instruction_tuning/12001.pt", map_location="cpu")
raw_model.load_state_dict(ckpt)
raw_model.eval()
model = raw_model.cuda()
print("ready")


def question_answer(prompt):
    print(prompt)
    raw_inputs = "user:{}<s>system:".format(prompt)
    inputs_len = len(raw_inputs)
    inputs = tokenizer(raw_inputs, return_tensors=True, add_special_tokens=False)
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    pred = model.generate(**inputs, max_new_tokens=512, do_sample=True)
    pred = tokenizer.decode(pred.cpu())[0]
    pred = pred[inputs_len:]
    print(pred)
    return pred


demo = gr.Interface(fn=question_answer, inputs="text", outputs="text").queue(
    concurrency_count=1
)
demo.launch(share=True)
