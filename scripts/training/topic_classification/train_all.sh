#!/bin/bash

python -m scripts.training.topic_classification.sgd

python -m scripts.training.topic_classification.bert_lin_text --model cointegrated/rubert-tiny2
python -m scripts.training.topic_classification.bert_lin_text --model DeepPavlov/rubert-base-cased --tokenizer_max_length 512
python -m scripts.training.topic_classification.bert_lin_text --model ai-forever/ruBert-large

python -m scripts.training.topic_classification.bert_mlp_text --model cointegrated/rubert-tiny2
python -m scripts.training.topic_classification.bert_mlp_text --model DeepPavlov/rubert-base-cased --tokenizer_max_length 512
python -m scripts.training.topic_classification.bert_mlp_text --model ai-forever/ruBert-large

python -m scripts.training.topic_classification.bert_multifeature.py --model cointegrated/rubert-tiny2
python -m scripts.training.topic_classification.bert_multifeature.py --model DeepPavlov/rubert-base-cased --tokenizer_max_length 512
