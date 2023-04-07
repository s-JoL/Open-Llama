"""
Author: LiangSong(sl12160010@gmail.com)
Date: 2023-04-06 22:30:10
LastEditors: LiangSong(sl12160010@gmail.com)
LastEditTime: 2023-04-07 23:03:31
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
    "data/saved_ckpt/instruction_tuning_math_code_multiturn/36001.pt",
    map_location="cpu",
)
raw_model.load_state_dict(ckpt)
raw_model.eval()
model = raw_model.cuda()
print("ready")


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = "</code></pre>"
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # [Open-Llama](https://github.com/Bayes-Song/Open-Llama)
    完全使用Open-Llama项目从0开始训练的Instruct-GPT模型，当长时间无响应（如20s以上）可刷新重试。

    Instruct-GPT model is trained from scratch using the Open-Llama project without relying on any other pre-trained models. If there is no response for a long time (such as more than 20 seconds), please refresh and try again. 
    """
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        print(user_message)
        return "", history + [[user_message, None]]

    def bot(history):
        context = []
        round = 0
        for prompt, completion in history:
            round += 1
            if completion is None:
                inputs = "user:{}\nsystem:".format(prompt)
                inputs = tokenizer(
                    inputs, return_tensors=True, add_special_tokens=False
                )
                context.append(inputs["input_ids"])
            else:
                inputs = "user:{}\nsystem:{}".format(prompt, completion)
                inputs = tokenizer(inputs, return_tensors=True, add_special_tokens=True)
                context.append(inputs["input_ids"])
        context = torch.cat(context, dim=-1)
        context = context[:, -1024:]
        inputs_len = context.shape[1]
        context = context.cuda()
        pred = model.generate(input_ids=context, max_new_tokens=512, do_sample=True)
        pred = pred[:, inputs_len:]
        pred = tokenizer.decode(pred.cpu())[0]
        print(pred)
        bot_message = parse_codeblock(pred)
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    gr.Markdown(
        """
    当前体验服务生成的所有内容都是由人工智能模型生成，我们对其生成内容的准确性、完整性和功能性不做任何保证，并且其生成的内容不代表我们的态度或观点。

    联系方式: sl12160010@gmail.com  对于该项目有任何意见和建议都欢迎联系我.
    Contact information: sl12160010@gmail.com. Any opinions or suggestions regarding the project are welcome to be addressed to me through this email.
    """
    )

demo.launch(share=True)
