import torch
import pandas as pd
from models import RuBERTMultifeature
from trainer import CustomTrainer, FocalLoss
from transformers import AutoTokenizer
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser(description='Train a linear model on top of RuBERT')
parser.add_argument('--model', type=str, help='Model name', default='cointegrated/rubert-tiny2')
parser.add_argument('--tokenizer_max_length', type=int, help='Tokenizer max length', default=None)
args = parser.parse_args()

train = pd.read_json('./data/labeled/train.jsonl', lines=True)
test = pd.read_json('./data/labeled/test.jsonl', lines=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.tokenizer_max_length:
    tokenizer.model_max_length = args.tokenizer_max_length

y_train = train['topic'].astype('category').cat.codes
y_test = test['topic'].astype('category').cat.codes

train_text_encodings = tokenizer(train['text'].tolist(), truncation=True, padding=True)
train_title_encodings = tokenizer(train['title'].tolist(), truncation=True, padding=True)
train_keywords_encodings = tokenizer(train['keywords'].tolist(), truncation=True, padding=True)
train_ner_encodings = tokenizer(train['ner'].tolist(), truncation=True, padding=True)

test_text_encodings = tokenizer(test['text'].tolist(), truncation=True, padding=True)
test_title_encodings = tokenizer(test['title'].tolist(), truncation=True, padding=True)
test_keywords_encodings = tokenizer(test['keywords'].tolist(), truncation=True, padding=True)
test_ner_encodings = tokenizer(test['ner'].tolist(), truncation=True, padding=True)

train_dataset = Dataset.from_dict({
    'input_ids': train_text_encodings['input_ids'],
    'attention_mask': train_text_encodings['attention_mask'],
    'title_input_ids': train_title_encodings['input_ids'],
    'title_attention_mask': train_title_encodings['attention_mask'],
    'keywords_input_ids': train_keywords_encodings['input_ids'],
    'keywords_attention_mask': train_keywords_encodings['attention_mask'],
    'ner_input_ids': train_ner_encodings['input_ids'],
    'ner_attention_mask': train_ner_encodings['attention_mask'],
    'labels': y_train.tolist()
})

test_dataset = Dataset.from_dict({
    'input_ids': test_text_encodings['input_ids'],
    'attention_mask': test_text_encodings['attention_mask'],
    'title_input_ids': test_title_encodings['input_ids'],
    'title_attention_mask': test_title_encodings['attention_mask'],
    'keywords_input_ids': test_keywords_encodings['input_ids'],
    'keywords_attention_mask': test_keywords_encodings['attention_mask'],
    'ner_input_ids': test_ner_encodings['input_ids'],
    'ner_attention_mask': test_ner_encodings['attention_mask'],
    'labels': y_test.tolist()
})

model = RuBERTMultifeature(args.model, num_classes=train['topic'].nunique())

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=FocalLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)
)

trainer.train(2)

trainer.plot_training_history()

torch.save(model, f'./model/saved/bert_multifeature_{args.model}.pt')