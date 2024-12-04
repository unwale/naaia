# Models

Contiains model classes for further interaction (evaluation). 
Each model class provides a `predict` method that takes a list of texts and returns a list of predictions.

## Available models
### Topic classification
* `RubertLinText` -- `DeepPavlov/rubert-base-cased` with linear head
* `RubertTinyLinText` -- `cointegrated/rubert-tiny2` with linear head
* `RubertMlpText` -- `DeepPavlov/rubert-base-cased` with MLP head
* `RubertTinyMlpText` -- `cointegrated/rubert-tiny2` with MLP head
* `RubertMultifeature` -- `DeepPavlov/rubert-base-cased` with multi-feature head
* `RubertTinyMultifeature` -- `cointegrated/rubert-tiny2` with multi-feature head
* `YandexZeroshot` -- YandexGPT Classification API for zero-shot classification
* `YandexFewshot` -- YandexGPT Classification API for few-shot classification
* `YandexFinetuned` -- YandexGPT Foundation Model fine-tuned on topic classification
* `GigaChatZeroShot` -- GigaChat model for zero-shot classification (GigaChat-Pro available as well)
* `GigaChatFewShot` -- GigaChat model for few-shot classification (GigaChat-Pro available as well)
## Topic ranking
* `GigaChatRanking` -- GigaChat Completions API for ranking
* `YandexRanking` -- YandexGPT Classification API (confidence scores are used for ranking)