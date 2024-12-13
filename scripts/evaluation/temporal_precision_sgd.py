import csv

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = pd.read_json("./data/time_precision/data.jsonl", lines=True)

train, test = train_test_split(data, test_size=0.2, random_state=42)

model = joblib.load("./model/saved/time_precision_model.joblib")

predictions = model.predict(test["context"])

report = classification_report(
    test["precision"], predictions, output_dict=True
)

with open("./results/time_precision_classification_report.csv", "w") as f:
    scores = [
        val["f1-score"] if isinstance(val, dict) else val
        for val in report.values()
    ]
    writer = csv.writer(f)
    writer.writerow(report.keys())
    writer.writerow(scores)
