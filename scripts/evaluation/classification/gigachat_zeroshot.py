import argparse

import pandas as pd
from inference.gigachat import GigaChatZeroshot
from sklearn.metrics import classification_report
from utils import save_classification_report

parser = argparse.ArgumentParser(
    description="Evaluate Gigachat Zeroshot model"
)
parser.add_argument("--model", type=str, help="Model name", default="GigaChat")
args = parser.parse_args()

model = GigaChatZeroshot(args.model)

test_data = pd.read_json("./data/labeled/test.jsonl", lines=True).sample(250)
topics = test_data["topic"].unique().tolist()

test_data["title"] = test_data["title"].fillna(
    test_data["text"].apply(lambda x: x.split("\n")[0])
)
test_data["title"] = test_data["title"].apply(lambda x: " ".join(x.split()))

predictions = model.predict(test_data["title"].tolist(), topics=topics)
true_labels = test_data["topic"].tolist()

report = classification_report(true_labels, predictions, output_dict=True)
save_classification_report(report, f"{args.model}-Zeroshot-Title")
