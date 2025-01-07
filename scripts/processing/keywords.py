import argparse
import json

import nltk
import numpy as np
import pandas as pd
import pytextrank  # noqa: F401
import spacy
from rake_nltk import Rake
from tqdm import tqdm
from yake import KeywordExtractor

# python -m spacy download ru_core_news_sm
spc = spacy.load("ru_core_news_sm")

nltk.download("stopwords")
nltk.download("punkt_tab")
stopwords = nltk.corpus.stopwords.words("russian")
_doc = spc(" ".join(stopwords))
stopwords = {w.text: [w.pos_] for w in _doc}

spc.add_pipe("textrank", config={"stopwords": stopwords})
yake = KeywordExtractor(
    lan="ru", n=2, top=15, stopwords=nltk.corpus.stopwords.words("russian")
)
rake = Rake(
    max_length=2,
    language="russian",
    stopwords=nltk.corpus.stopwords.words("russian"),
)


def get_keywords_yake(text):
    keywords = yake.extract_keywords(text)
    if not keywords:
        return []
    # normalize scores
    scores = np.array([kw[1] for kw in keywords])
    scores = scores - scores.max()
    scores = 1 - scores
    return sorted(
        [(keyword[0], score) for keyword, score in zip(keywords, scores)],
        key=lambda x: x[1],
        reverse=True,
    )


def get_keywords_rake(text):
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases_with_scores()
    return [(kw[1], kw[0]) for kw in keywords]


def get_keywords_textrank(text):
    doc = spc(text)
    keywords = {phrase.text: phrase.rank for phrase in doc._.phrases[:10]}
    if not keywords:
        return []
    # normalize scores
    scores = np.array(list(keywords.values()))
    if scores.max() > 0:
        scores = scores / scores.max()
    return [
        (keyword, score) for keyword, score in zip(keywords.keys(), scores)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keywords from text")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input jsonl file",
        default=".../data/preprocessed/test.jsonl",
    )
    args = parser.parse_args()

    data = pd.read_json(args.input, lines=True)
    data["yake_keywords"] = None
    data["rake_keywords"] = None
    data["textrank_keywords"] = None

    for i, text in tqdm(data["text"].items(), total=len(data)):
        yake_keywords = get_keywords_yake(text)
        rake_keywords = get_keywords_rake(text)
        textrank_keywords = get_keywords_textrank(text)

        data.at[i, "yake_keywords"] = (
            json.dumps(yake_keywords) if yake_keywords else None
        )
        data.at[i, "rake_keywords"] = (
            json.dumps(rake_keywords) if rake_keywords else None
        )
        data.at[i, "textrank_keywords"] = (
            json.dumps(textrank_keywords) if textrank_keywords else None
        )

    file_name = args.input.split("/")[-1]
    output_path = f"./data/processed/{file_name}"

    data.to_json(output_path, lines=True, orient="records")
