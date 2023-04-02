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
ckpt = torch.load(
    "data/saved_ckpt/instruction_tuning/14001.pt",
    map_location="cpu",
)
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


demo = gr.Interface(
    fn=question_answer,
    inputs="text",
    outputs="text",
    examples=[
        "帮我写一封邮件，内容是感谢jack的帮助，希望有机会能和他线下见面，请他吃饭",
        "情人节送女朋友什么礼物，预算500",
        "我今天肚子有点不舒服，晚饭有什么建议么",
        "可以总结一下小说三体的核心内容么？",
        "Can you explain to me what quantum mechanics is and how it relates to quantum computing?",
        "I'm feeling a bit unwell in my stomach today. Do you have any suggestions for dinner?",
    ],
).queue(
    concurrency_count=1,
    title="Open-Llama",
    description="不基于其他预训练模型，完全使用Open-Llama项目从0开始训练的Instruct-GPT模型，总共训练花费在2w美元以内，训练时间80h。",
    article="[关于作者](http://home.ustc.edu.cn/~sl9292/resume.html)",
)
demo.launch(share=True)
