#!/bin/bash

# bert-based models
python -m scripts.evaluation.classification.bert --model ./model/saved/bert_mlp_text_tiny.pth --tokenizer cointegrated/rubert-tiny2

# gigachat
python scripts/evaluation/gigachat_zeroshot.py --model GigaChat
python scripts/evaluation/gigachat_zeroshot.py --model GigaChat-Pro

# yandex-gpt
python scripts/evaluation/yandex_zero_shot.py
python scripts/evaluation/yandex_few_shot.py