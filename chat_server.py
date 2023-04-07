"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-04-06 22:30:10
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-06 23:13:54
FilePath: /Open-Llama/chat_server.py
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
ckpt = torch.load(
    "data/saved_ckpt/instruction_tuning_3_epochs/37001.pt", map_location="cpu"
)
raw_model.load_state_dict(ckpt)
raw_model.eval()
model = raw_model.cuda()
print("ready")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        context = []
        round = 0
        for prompt, completion in history:
            round += 1
            if completion is None:
                inputs = 'user:{}\nsystem:'.format(prompt)
                inputs = tokenizer(inputs, return_tensors=True, add_special_tokens=False)
                context.append(inputs['input_ids'])
            else:
                inputs = 'user:{}\nsystem:{}'.format(prompt, completion)
                inputs = tokenizer(inputs, return_tensors=True, add_special_tokens=True)
                context.append(inputs['input_ids'])
        context = torch.cat(context, dim=-1)
        context = context[:, -1024: ]
        inputs_len = context.shape[1]
        context = context.cuda()
        pred = model.generate(input_ids=context, max_new_tokens=512, do_sample=True)
        pred = pred[:, inputs_len:]
        pred = tokenizer.decode(pred.cpu())[0]
        bot_message = pred
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
