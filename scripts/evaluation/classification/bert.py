import argparse

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from scripts.evaluation.classification.utils import save_classification_report
from scripts.evaluation.inference.bert import BertTextClassifier

load_dotenv()

parser = argparse.ArgumentParser(
    description="Evaluate a linear model on top of RuBERT"
)
parser.add_argument("--model_path", type=str, help="Model path")
parser.add_argument("--base_model", type=str, help="Base model name")
parser.add_argument("--tokenizer", type=str, help="Tokenizer path")
parser.add_argument(
    "--tokenizer_max_length",
    type=int,
    help="Tokenizer max length",
    default=None,
)
args = parser.parse_args()

model_name = args.model_path.split("/")[-1].replace(".pth", "")
model = BertTextClassifier(
    args.model_path, args.base_model_name, args.tokenizer
)
if args.tokenizer_max_length:
    model.tokenizer.model_max_length = args.tokenizer_max_length

test = pd.read_json("./data/labeled/test.jsonl", lines=True)
train = pd.read_json("./data/labeled/train.jsonl", lines=True)

texts = test["text"].tolist()
topics = (
    train["topic"].unique().tolist()
)  # TODO figure out the correct order of topics OR retrain all models

gt = test["topic"].tolist()

predictions = model.predict(texts, topics)
report = classification_report(gt, predictions, output_dict=True)
save_classification_report(report, model_name)
