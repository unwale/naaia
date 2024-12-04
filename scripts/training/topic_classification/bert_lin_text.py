import torch
import pandas as pd
from models import RuBERTLinText
from trainer import CustomTrainer, FocalLoss
from transformers import AutoTokenizer
from datasets import Dataset
from utils import plot_training_history
from matplotlib import pyplot as plt 
import argparse

parser = argparse.ArgumentParser(description='Train a linear model on top of RuBERT')
parser.add_argument('--model', type=str, help='Model name', default='cointegrated/rubert-tiny2')
parser.add_argument('--tokenizer_max_length', type=int, help='Tokenizer max length', default=None)
args = parser.parse_args()

# Load data
train = pd.read_json('./data/labeled/train.jsonl', lines=True)
test = pd.read_json('./data/labeled/test.jsonl', lines=True)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.tokenizer_max_length:
    tokenizer.model_max_length = args.tokenizer_max_length

train_encodings = tokenizer(train['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test['text'].tolist(), truncation=True, padding=True)
y_train = train['topic'].astype('category').cat.codes
y_test = test['topic'].astype('category').cat.codes

# Create datasets
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train.tolist()
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test.tolist()
})

model = RuBERTLinText(args.model, num_classes=train['topic'].nunique())

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=FocalLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)
)

trainer.train(4)

trainer.plot_training_history()

torch.save(model.bert.state_dict(), "./model/saved/tiny_bert.pth")
