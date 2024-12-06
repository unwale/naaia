import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from utils import make_batch, parse_batch_results, watch_batch

load_dotenv()


parser = argparse.ArgumentParser(
    description="Generate user-messages using OpenAI Batch API"
)
parser.add_argument(
    "--model", type=str, help="Model name", default="gpt-4o-mini"
)
parser.add_argument(
    "--input",
    type=str,
    help="Path to joblib file of a pandas table",
    default="../data/processed/data_keywords_ent_ranked.pkl",
)
parser.add_argument(
    "--output",
    type=str,
    help="Path to save the output",
    default="../data/batch/users_messages_on_text.json",
)
args = parser.parse_args()

topics = [
    "Учебный процесс",
    "Научные исследования",
    "Студенческая жизнь",
    "Социальные инициативы",
    "Инфраструктура и услуги",
    "Проблемы и вызовы",
    "Новости администрации",
    "Трудоустройство и карьера",
    "Культура и искусство",
    "Спорт",
    "Актуальные события",
    "Здоровье и благополучие",
    "Другое",
]

openai = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

data = pd.read_json(args.input, lines=True)

response_schema = {
    "name": "UserMessageGenerationResult",
    "schema": {
        "type": "object",
        "properties": {
            "user_messages": {"type": "array", "items": {"type": "string"}}
        },
    },
}


batch_size = 3000
batches = [
    data.iloc[i : i + batch_size] for i in range(0, len(data), batch_size)
]

topics = [
    "Учебный процесс",
    "Научные исследования",
    "Студенческая жизнь",
    "Социальные инициативы",
    "Инфраструктура и услуги",
    "Проблемы и вызовы",
    "Новости администрации",
    "Трудоустройство и карьера",
    "Культура и искусство",
    "Спорт",
    "Актуальные события",
    "Здоровье и благополучие",
    "Другое",
]

data["queries"] = None
prompt = (
    "напиши три независимых запроса ии-ассистенту, которые мог бы "
    + "отправить пользователь, который хочет узнать о чем-то, что "
    + 'упоминается в этом тексте: "{text}". сообщение должно быть '
    + "коротким и быть написанным в разговорном стиле. отвечай в "
    + "формате JSON объекта с полем user_messages."
)

for i, batch in enumerate(batches):
    print(f"Batch {i} started")
    prompts = []
    for text in batch["text"].values:
        text_prompt = prompt.format(text=" ".join(text.split()))
        prompts.append(text_prompt)
    prompts = pd.Series(prompts)
    batch_id = make_batch(openai, args.model, prompts, response_schema)
    results = watch_batch(openai, batch_id, f"{args.output[:-6]}{i}.jsonl")
    queries = parse_batch_results(results)
    data.loc[batch.index, "queries"] = queries
    print(f"Batch {i} done")

data["queries"] = data["queries"].apply(
    lambda x: json.loads(x)["user_messages"]
)
data["queries"] = data["queries"].apply(lambda x: json.dumps(x))

data[["text", "title", "queries", "topic_ranking"]].to_json(
    args.output, orient="records", lines=True
)
