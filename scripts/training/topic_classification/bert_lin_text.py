import argparse

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from scripts.training.topic_classification.models import RuBERTLinText
from scripts.training.topic_classification.trainer import (
    CustomTrainer,
    FocalLoss,
)

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

model_name = args.model.split("/")[-1] + "-lin"

train = pd.read_json("./data/labeled/train.jsonl", lines=True)
test = pd.read_json("./data/labeled/test.jsonl", lines=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.tokenizer_max_length:
    tokenizer.model_max_length = args.tokenizer_max_length

train_encodings = tokenizer(
    train["text"].tolist(), truncation=True, padding=True
)
test_encodings = tokenizer(
    test["text"].tolist(), truncation=True, padding=True
)
y_train = train["topic"].astype("category").cat.codes
y_test = test["topic"].astype("category").cat.codes

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

model = RuBERTLinText(args.model, num_classes=train["topic"].nunique())

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=FocalLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
)

trainer.train(2)

trainer.plot_training_history()

torch.save(model.state_dict(), f"./model/saved/{model_name}.pth")
