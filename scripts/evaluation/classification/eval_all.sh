#!/bin/bash

python -m scripts.evaluation.classification.sgd

# bert
python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/rubert-tiny2-lin.pth \
    --base_model cointegrated/rubert-tiny2 \
    --tokenizer cointegrated/rubert-tiny2
python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/rubert-base-cased-lin.pth \
    --base_model DeepPavlov/rubert-base-cased \
    --tokenizer DeepPavlov/rubert-base-cased \
    --tokenizer_max_length 512
python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/ruBert-large-lin.pth \
    --base_model ai-forever/ruBert-large \
    --tokenizer ai-forever/ruBert-large \
    --tokenizer_max_length 512

python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/rubert-tiny2-mlp.pth \
    --base_model cointegrated/rubert-tiny2 \
    --tokenizer cointegrated/rubert-tiny2
python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/rubert-base-cased-mlp.pth \
    --base_model DeepPavlov/rubert-base-cased \
    --tokenizer DeepPavlov/rubert-base-cased \
    --tokenizer_max_length 512
python -m scripts.evaluation.classification.bert \
    --model_path ./model/saved/ruBert-large-mlp.pth \
    --base_model ai-forever/ruBert-large \
    --tokenizer ai-forever/ruBert-large \
    --tokenizer_max_length 512

# gigachat
python -m scripts.evaluation.classification.gigachat_zeroshot --model GigaChat
python -m scripts.evaluation.classification.gigachat_zeroshot --model GigaChat-Pro

# yandex-gpt
python -m scripts.evaluation.classification.yandex_zero_shot
python -m scripts.evaluation.classification.yandex_few_shot
