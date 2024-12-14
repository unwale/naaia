#!/bin/bash

python -m scripts.evaluation.classification.sgd

# bert-based models
# TODO add bert-based models

# gigachat
python -m scripts.evaluation.classification.gigachat_zeroshot --model GigaChat
python -m scripts.evaluation.classification.gigachat_zeroshot --model GigaChat-Pro

# yandex-gpt
python -m scripts.evaluation.classification.yandex_zero_shot
python -m scripts.evaluation.classification.yandex_few_shot
