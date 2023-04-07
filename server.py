"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-03-31 13:26:15
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-06 03:45:44
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
    "data/saved_ckpt/instruction_tuning_3_epochs/37001.pt", map_location="cpu"
)
raw_model.load_state_dict(ckpt)
raw_model.eval()
model = raw_model.cuda()
print("ready")


def question_answer(prompt):
    print(prompt)
    raw_inputs = "user:{}\nsystem:".format(prompt)
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
        "帮我写一封邮件，内容是咨询教授本学期量子力学课程的时间表？并且希望教授推荐一些相关书籍",
        "情人节送女朋友什么礼物，预算500",
        "我今天肚子有点不舒服，晚饭有什么建议么",
        "可以总结一下小说三体的核心内容么？",
        "Can you explain to me what quantum mechanics is and how it relates to quantum computing?",
        "请帮我写一个AI驱动的幼儿教育APP的商业计划书",
        "用python实现一个快速排序",
    ],
    title="Open-Llama",
    description="不基于其他预训练模型，完全使用[Open-Llama](https://github.com/Bayes-Song/Open-Llama)项目从0开始训练的Instruct-GPT模型，总训练成本不超过2w美元。由于请求需要经Gradio进行转发，可能出现请求丢失的现象，当长时间无响应（如20s以上）可刷新重试。当前体验服务生成的所有内容都是由人工智能模型生成，我们对其生成内容的准确性、完整性和功能性不做任何保证，并且其生成的内容不代表我们的态度或观点。",
    article="联系方式: sl12160010@gmail.com  对于该项目有任何意见和建议都欢迎联系我",
).queue(concurrency_count=1)
demo.launch(share=True)
