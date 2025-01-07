import argparse
import logging
import re
import string

import nltk
import pandas as pd
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")

parser = argparse.ArgumentParser(
    description="Clean and lemmatize text in a jsonl file"
)
parser.add_argument(
    "--input",
    type=str,
    help="Path to input jsonl file",
    default=".../data/raw/data.jsonl",
)
parser.add_argument(
    "--target-field",
    type=str,
    help="Name of the field to clean and lemmatize",
    default="text",
)
parser.add_argument(
    "--output",
    type=str,
    help="Path to save the output",
    default=".../data/processed/data.jsonl",
)

logger.info("Loading spacy model and nltk stopwords")
nlp = spacy.load("ru_core_news_sm")
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words("russian"))


def clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"\[.*?\|([^]]+)\]", r"\1", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the text using spacy.
    """
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info("Reading data")
    data = pd.read_json(args.input, lines=True)
    data = data.dropna(subset=[args.target_field])

    logger.info(f"Cleaning {args.target_field}")
    data["cleaned_text"] = data[args.target_field].apply(clean_text)

    logger.info("Lemmatizing text")
    data["lemmatized_text"] = data["cleaned_text"].apply(lemmatize_text)
    data.drop(columns=["cleaned_text"], inplace=True)
    data = data[data["lemmatized_text"] != ""]

    data.to_json(args.output, orient="records", lines=True)
    logger.info(f"Data saved to {args.output}")
