#!/bin/bash
###
 # @Author: s-JoL(sl12160010@gmail.com)
 # @Date: 2023-03-16 21:21:56
 # @LastEditors: s-JoL(sl12160010@gmail.com)
 # @LastEditTime: 2023-03-26 22:58:11
 # @FilePath: /Open-Llama/data/download_wudao.sh
 # @Description: 
 # download wudao dataset and preprocess
 # Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
### 
apt install unrar

wget -v -c 'https://download.scidb.cn/download?fileId=63a30383fed6a8a9e8454302&dataSetType=organization&fileName=WuDaoCorporaText-2.0-open.rar' -O data/WuDaoCorpus2.0_base_200G.rar

# for i in {1..100}
# do
#   curl -C - --retry 100 'https://dorc.baai.ac.cn/resources/data/WuDaoCorpora2.0/WuDaoCorpus2.0_base_200G.rar?AccessKeyId=AKLTNasiLRBBTcOgPqzlkPzu1w&Expires=1679127659&Signature=7jh%2FpnJyC2hAeumm9EjaeE5HN9E%3D' -o data/WuDaoCorpus2.0_base_200G.rar
# done

unrar x data/WuDaoCorpus2.0_base_200G.rar data/
mkdir data/pretrain_data
python3 data/preprocess_wudao.py