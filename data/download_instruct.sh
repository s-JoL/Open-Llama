#!/bin/bash
###
 # @Author: s-JoL(sl12160010@gmail.com)
 # @Date: 2023-04-05 23:18:10
 # @LastEditors: s-JoL(sl12160010@gmail.com)
 # @LastEditTime: 2023-05-04 08:24:17
 # @FilePath: /Open-Llama/data/download_instruct.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
### 
mkdir data/instruction_data
wget -c --tries 3 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json' -O data/sg_90k_part1_html_cleaned.json
wget -c --tries 3 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json' -O data/sg_90k_part2_html_cleaned.json
python3 data/preprocess_instruction.py