###
 # @Author: LiangSong(sl12160010@gmail.com)
 # @Date: 2023-04-08 22:44:27
 # @LastEditors: LiangSong(sl12160010@gmail.com)
 # @LastEditTime: 2023-04-11 21:58:43
 # @FilePath: /Open-Llama/speed_test/accelerate/run.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
### 
total_gpu=8
accelerate launch --config_file deepspeed_stage2.yaml --main_process_ip 127.0.0.1 --main_process_port 23335 --num_processes $total_gpu run.py