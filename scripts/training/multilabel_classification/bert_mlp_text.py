import argparse
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

from scripts import topics
from scripts.training.topic_classification.models import RuBERTMLPText
from scripts.training.trainer import CustomTrainer, MultilabelFocalLoss

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

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.tokenizer_max_length:
    tokenizer.model_max_length = args.tokenizer_max_length

threshold = 80
train = pd.read_json("./data/ranked/test.jsonl", lines=True)
test = pd.read_json("./data/ranked/test.jsonl", lines=True)
train["topic_ranking"] = train["topic_ranking"].apply(json.loads)
test["topic_ranking"] = test["topic_ranking"].apply(json.loads)
train["topic_list"] = train["topic_ranking"].apply(
    lambda x: [t for t in topics if x[t] >= threshold]
)
test["topic_list"] = test["topic_ranking"].apply(
    lambda x: [t for t in topics if x[t] >= threshold]
)

mlb = MultiLabelBinarizer(classes=topics)
y_train = mlb.fit_transform(train["topic_list"])
y_test = mlb.transform(test["topic_list"])

train_encodings = tokenizer(
    train["text"].tolist(), truncation=True, padding=True
)
test_encodings = tokenizer(
    test["text"].tolist(), truncation=True, padding=True
)

train_dataset = Dataset.from_dict(
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": y_train,
    }
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": y_test,
    }
)

model = RuBERTMLPText(args.model, num_classes=len(topics), hidden_size=128)

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=MultilabelFocalLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
)

trainer.train(2)

trainer.plot_training_history()

torch.save(model.state_dict(), f"./model/saved/{model_name}.pth")
