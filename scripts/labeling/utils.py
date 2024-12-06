import json
import time
from io import BytesIO
from typing import List

import pandas as pd
from openai import OpenAI


def make_batch(
    openai: OpenAI, model: str, prompts: pd.Series, schema=None, temperature=0
):
    batch = []
    for index, text in prompts.items():
        text = " ".join(text.split())
        request = {
            "custom_id": f"r_{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "temperature": temperature,
                "n": 1,
            },
        }

        if schema:
            request["body"]["response_format"] = {
                "type": "json_schema",
                "json_schema": schema,
            }

        batch.append(json.dumps(request, ensure_ascii=False))

    buffer = BytesIO()
    buffer.write("\n".join(batch).encode())

    file_id = openai.files.create(file=buffer, purpose="batch").id

    batch = openai.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch.id


def watch_batch(openai: OpenAI, batch_id: str, path: str):
    prev_status = None
    batch = None
    while True:
        batch = openai.batches.retrieve(batch_id)
        status = batch.status

        if status != prev_status:
            print(f"Batch status: {status}")
            prev_status = status

        if status == "completed":
            break

        time.sleep(10)

    file_id = batch.output_file_id

    buffer = BytesIO()
    buffer.write(openai.files.content(file_id).content)
    with open(path, "wb") as f:
        f.write(openai.files.content(file_id).content)
    buffer.seek(0)

    return buffer.readlines()


def parse_batch_results(results: List[str]) -> pd.DataFrame:
    data = []
    for line in results:
        json_response = json.loads(line)
        data.append(
            json_response["response"]["body"]["choices"][0]["message"][
                "content"
            ]
        )

    return data
