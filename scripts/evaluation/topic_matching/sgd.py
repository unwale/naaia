import argparse
import json

import pandas as pd
from tqdm import tqdm

from model.inference.sgd import SGDClassifier
from scripts import topics

parser = argparse.ArgumentParser(
    description="Evaluate Gigachat Zeroshot model"
)
parser.add_argument("--model", type=str, help="Model name", default="GigaChat")
args = parser.parse_args()

model = SGDClassifier("model/saved/sgd.joblib")

text_queries = pd.read_json("./data/queries/queries.jsonl", lines=True)
text_queries["queries"] = text_queries["queries"].apply(json.loads)

match_count = 0
num_queries = 0
for i, row in tqdm(
    text_queries.iterrows(),
    desc="Predicting topics",
    total=text_queries.shape[0],
):
    text = row["text"]
    queries = row["queries"]
    text_topics = model.predict([text], topics)
    query_topics = model.predict(queries, topics)
    for i in range(len(queries)):
        num_queries += 1
        if text_topics[0] == query_topics[i]:
            match_count += 1

print(f"Topic matching accuracy: {match_count / num_queries:.2f}")

with open("./results/matching_metrics.csv", "a") as f:
    if f.tell() == 0:
        f.write("model,accuracy\n")
    f.write(f"sgd,{match_count / num_queries:.2f}\n")
