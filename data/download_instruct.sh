#!/bin/bash
###
 # @Author: LiangSong(sl12160010@gmail.com)
 # @Date: 2023-04-05 23:18:10
 # @LastEditors: LiangSong(sl12160010@gmail.com)
 # @LastEditTime: 2023-04-05 23:34:30
 # @FilePath: /Open-Llama/data/download_instruct.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
### 
mkdir data/instruction_data
curl -C - --retry 3 'https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json' -o data/sg_90k_part1.json
curl -C - --retry 3 'https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part2.json' -o data/sg_90k_part2.json
python3 data/preprocess_instruction.py