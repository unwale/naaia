import json
import pandas as pd
from tqdm import tqdm
from scripts.evaluation.inference.bert import BertTextClassifier

# TODO add argparse for model path & tokenizer

model = BertTextClassifier('/home/unwale/Projects/naaia/model/saved/bert_mlp_text_lg.pth', 'ai-forever/ruBert-large')

text_queries = pd.read_json('./data/queries/queries.jsonl', lines=True)
text_queries['queries'] = text_queries['queries'].apply(json.loads)

topics = [i for i in range(14)]
match_count = 0
num_queries = 0
for i, row in tqdm(text_queries.iterrows(), desc='Predicting topics', total=text_queries.shape[0]):
    text= row['text']
    queries = row['queries']
    text_topics = model.predict([text], topics)
    query_topics = model.predict(queries, topics)
    for i in range(len(queries)):
        num_queries += 1
        if text_topics[0] == query_topics[i]:
            match_count += 1

print(f'Query matching accuracy: {match_count / num_queries:.2f}')
