#!/bin/bash

python -m scripts.evaluation.topic_matching.sgd

# bert
python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/rubert-tiny2-lin.pth \
    --base_model cointegrated/rubert-tiny2 \
    --tokenizer cointegrated/rubert-tiny2
python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/rubert-base-cased-lin.pth \
    --base_model DeepPavlov/rubert-base-cased \
    --tokenizer DeepPavlov/rubert-base-cased \
    --tokenizer_max_length 512
python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/ruBert-large-lin.pth \
    --base_model ai-forever/ruBert-large \
    --tokenizer ai-forever/ruBert-large \
    --tokenizer_max_length 512

python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/rubert-tiny2-mlp.pth \
    --base_model cointegrated/rubert-tiny2 \
    --tokenizer cointegrated/rubert-tiny2
python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/rubert-base-cased-mlp.pth \
    --base_model DeepPavlov/rubert-base-cased \
    --tokenizer DeepPavlov/rubert-base-cased \
    --tokenizer_max_length 512
python -m scripts.evaluation.topic_matching.bert \
    --model_path ./model/saved/ruBert-large-mlp.pth \
    --base_model ai-forever/ruBert-large \
    --tokenizer ai-forever/ruBert-large \
    --tokenizer_max_length 512

# gigachat
python -m scripts.evaluation.topic_matching.gigachat --model GigaChat
python -m scripts.evaluation.topic_matching.gigachat --model GigaChat-Pro
