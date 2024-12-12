import argparse
import json
import re

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    help="Path to input jsonl file",
    default=".../data/raw/data.jsonl",
)
args = parser.parse_args()

data = pd.read_json(args.input, lines=True)

social_media_index = data[data["source_platform"].isin("VK", "TELEGRAM")].index

data.loc[social_media_index, "title"] = data.loc[
    social_media_index, "text"
].apply(lambda x: x.split("\n")[0])


tag_pattern = re.compile(r"#\w+")

data.loc[social_media_index, "tags"] = data.loc[
    social_media_index, "text"
].apply(lambda x: json.dumps(tag_pattern.findall(x)))

data.to_json(args.input, lines=True, orient="records")
