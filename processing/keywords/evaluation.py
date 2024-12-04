import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from keywords import get_keywords, get_keywords_rake, get_keywords_textrank, get_keywords_yake

load_dotenv()

# computes how similar are the texts and correcponding keywords
def avg_bert_similarity(texts, keywords):
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    text_embeddings = model.encode(texts)
    keyword_embeddings = model.encode(keywords)
    similarities = []
    for text_embedding, keyword_embedding in zip(text_embeddings, keyword_embeddings):
        similarities.append(np.dot(text_embedding, keyword_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(keyword_embedding)))
    
    return np.mean(similarities)

# compare performance with openai gpt4omini
def avg_gpt_similarity(texts, keywords):
    openai = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    similarities = []
    for text, keyword in zip(texts, keywords):
        response = openai.completions.create(
            model="gpt-4o-mini",
            prompt=f"Оцени от 0 до 100, насколько хорошо ключевые слова характеризуют этот текст: {text}  {keyword}",
            max_tokens=100,
            n=1,
            stop=None
        )
        similarities.append(response['similarity'])

if __name__ == '__main__':
    ...
    # TODO call the functions above with the corresponding data and probably move to /scripts/evaluation/something