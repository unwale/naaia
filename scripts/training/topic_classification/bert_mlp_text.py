import argparse

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from scripts import topics
from scripts.training.topic_classification.models import RuBERTMLPText
from scripts.training.trainer import CustomTrainer, FocalLoss

parser = argparse.ArgumentParser(
    description="Train a linear model on top of RuBERT"
)
parser.add_argument(
    "--model", type=str, help="Model name", default="cointegrated/rubert-tiny2"
)
parser.add_argument(
    "--tokenizer_max_length",
    type=int,
    help="Tokenizer max length",
    default=None,
)
args = parser.parse_args()

model_name = args.model.split("/")[-1] + "-mlp"

train = pd.read_json("./data/labeled/train.jsonl", lines=True)
test = pd.read_json("./data/labeled/test.jsonl", lines=True)
train = train[train["topic"].isin(topics)]
test = test[test["topic"].isin(topics)]

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.tokenizer_max_length:
    tokenizer.model_max_length = args.tokenizer_max_length

train_encodings = tokenizer(
    train["text"].tolist(), truncation=True, padding=True
)
test_encodings = tokenizer(
    test["text"].tolist(), truncation=True, padding=True
)
y_train = train["topic"].apply(lambda x: topics.index(x))
y_test = test["topic"].apply(lambda x: topics.index(x))

train_dataset = Dataset.from_dict(
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": y_train.tolist(),
    }
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": y_test.tolist(),
    }
)

model = RuBERTMLPText(
    args.model, num_classes=train["topic"].nunique(), hidden_size=128
)

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=FocalLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
)

trainer.train(2)

torch.save(model.state_dict(), f"./model/saved/{model_name}.pth")
