import csv
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()


# computes how similar are the texts and correcponding keywords
def avg_bert_similarity(texts, keywords):
    model = SentenceTransformer("cointegrated/rubert-tiny2")
    print("Computing embeddings")
    text_embeddings = model.encode(texts)
    keyword_embeddings = model.encode(keywords)
    similarities = []
    for text_embedding, keyword_embedding in zip(
        text_embeddings, keyword_embeddings
    ):
        similarities.append(
            np.dot(text_embedding, keyword_embedding)
            / (
                np.linalg.norm(text_embedding)
                * np.linalg.norm(keyword_embedding)
            )
        )

    return np.mean(similarities)


# compare performance with openai gpt4omini
def avg_gpt_similarity(texts, keywords):
    openai = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    scores = []
    for text, keyword in tqdm(zip(texts, keywords), total=len(texts)):
        messages = [
            {
                "role": "user",
                "content": "Оцени от 0 до 100, насколько хорошо ключевые слова"
                + f" характеризуют этот текст: {text}"
                + f"\n\nКлючевые слова для оценки: {keyword}",
            }
        ]
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
            n=1,
            stop=None,
        )
        text = response.choices[0].message.content.strip()

        match = re.search(r"\b\d{1,3}\b", text)
        if match:
            score = int(match.group())
            scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":

    test_data = pd.read_json("./data/processed/train.jsonl", lines=True)
    test_data.dropna(inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_data = test_data.sample(500)
    print(f"Loaded {len(test_data)} samples")

    test_data["yake_keywords"] = test_data["yake_keywords"].apply(json.loads)
    test_data["rake_keywords"] = test_data["rake_keywords"].apply(json.loads)
    test_data["textrank_keywords"] = test_data["textrank_keywords"].apply(
        json.loads
    )

    test_data["yake_keywords"] = test_data["yake_keywords"].apply(
        lambda x: [keyword[0] for keyword in x]
    )
    test_data["rake_keywords"] = test_data["rake_keywords"].apply(
        lambda x: [keyword[0] for keyword in x]
    )
    test_data["textrank_keywords"] = test_data["textrank_keywords"].apply(
        lambda x: [keyword[0] for keyword in x]
    )

    test_data["yake_keywords"] = test_data["yake_keywords"].apply(", ".join)
    test_data["rake_keywords"] = test_data["rake_keywords"].apply(", ".join)
    test_data["textrank_keywords"] = test_data["textrank_keywords"].apply(
        ", ".join
    )

    yake_similarity = avg_bert_similarity(
        test_data["text"].tolist(), test_data["yake_keywords"].tolist()
    )
    rake_similarity = avg_bert_similarity(
        test_data["text"].tolist(), test_data["rake_keywords"].tolist()
    )
    textrank_similarity = avg_bert_similarity(
        test_data["text"].tolist(), test_data["textrank_keywords"].tolist()
    )

    print(f"YAKE embedding similarity: {yake_similarity}")
    print(f"RAKE embedding similarity: {rake_similarity}")
    print(f"TextRank embedding similarity: {textrank_similarity}")

    yake_gpt_similarity = avg_gpt_similarity(
        test_data["text"].tolist(), test_data["yake_keywords"].tolist()
    )
    rake_gpt_similarity = avg_gpt_similarity(
        test_data["text"].tolist(), test_data["rake_keywords"].tolist()
    )
    textrank_gpt_similarity = avg_gpt_similarity(
        test_data["text"].tolist(), test_data["textrank_keywords"].tolist()
    )

    print(f"YAKE GPT similarity: {yake_similarity}")
    print(f"RAKE GPT similarity: {rake_similarity}")
    print(f"TextRank GPT similarity: {textrank_similarity}")

    with open("./results/keywords_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Embedding Similarity", "GPT Similarity"])
        writer.writerow(["YAKE", yake_similarity, yake_gpt_similarity])
        writer.writerow(["RAKE", rake_similarity, rake_gpt_similarity])
        writer.writerow(
            ["TextRank", textrank_similarity, textrank_gpt_similarity]
        )
