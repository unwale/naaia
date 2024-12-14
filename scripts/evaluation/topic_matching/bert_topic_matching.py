import argparse
import json

import pandas as pd
from tqdm import tqdm

from model.inference.bert import BertTextClassifier

parser = argparse.ArgumentParser()
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

text_queries = pd.read_json("./data/queries/queries.jsonl", lines=True)
text_queries["queries"] = text_queries["queries"].apply(json.loads)

topics = [i for i in range(14)]
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
