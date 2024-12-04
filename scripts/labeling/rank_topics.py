import os
import pandas as pd
import json
import time
import pandas as pd
from openai import OpenAI
from utils import make_batch, watch_batch, parse_batch_results
from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser(description='Rank topics using OpenAI Batch API')
parser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini')
parser.add_argument('--path', type=str, help='Path to jsonl file to rank', default='./data/preprocessed/test.jsonl')
parser.add_argument('--column', type=str, help='Name of the column to rank topics by', default='text')
args = parser.parse_args()

openai = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

data = pd.read_json(args.path, lines=True)

response_schema = {
                    'name': 'TopicRankingResult',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'topics': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'topic': {
                                            'type': 'string'
                                        },
                                        'score': {
                                            'type': 'number'
                                        }
                                    }
                                }
                            }
                        }
                    }
                }      

topics = ['Учебный процесс',
        'Научные исследования',
        'Студенческая жизнь',
        'Социальные инициативы',
        'Инфраструктура и услуги',
        'Проблемы и вызовы',
        'Новости администрации',
        'Трудоустройство и карьера',
        'Культура и искусство',
        'Спорт',
        'Актуальные события',
        'Здоровье и благополучие',
        'Другое']

topics_list = ', '.join([topic for topic in topics])
prompt = "Оцени от 1 до 100, насколько каждая из предложенных тем релевантна для этого текста (у более релевантной темы должна быть более высокая оценка): \"{text}\". Темы, которые нужно оценить по релевантности: " + topics_list +". Отвечай в формате JSON списка из объектов с полями topic и score."

batch_size = 3000
batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]
data['topic_ranking'] = None

for i, batch in enumerate(batches):
    prompts = batch[args.column].apply(lambda x: prompt.format(text=' '.join(x.split())))
    
    batch_id = make_batch(openai, args.model, prompts, response_schema)
    results = watch_batch(openai, batch_id, f'./data/batch/output{i}.jsonl')
    data.loc[batch.index, 'topic_ranking'] = parse_batch_results(results)
    print(f'Batch {i} done')

# drop nan
data.loc[:, 'topic_ranking'] = data.loc[:, 'topic_ranking'].apply(json.loads)
data.loc[:, 'topic_ranking'] = data.loc[:, 'topic_ranking'].apply(lambda x: x['topics'])
data.loc[:, 'topic_ranking'] = data.loc[:, 'topic_ranking'].apply(lambda x: {item['topic']: item['score'] for item in x})
# check if all topics are present in each ranking
for i, ranking in data.loc[:, 'topic_ranking'].items():
    for topic in topics:
        if topic not in ranking:
            ranking[topic] = 0
    for topic in list(ranking.keys()):
        if topic not in topics:
            del ranking[topic]
# if all topics have score 0, set Другое to 100
for i, ranking in data.loc[:, 'topic_ranking'].items():
    if all(score == 0 for score in ranking.values()):
        ranking['Другое'] = 100
data.loc[:, 'topic_ranking'] = data.loc[:, 'topic_ranking'].apply(json.dumps)

filename = args.path.split('/')[-1] 
data.to_json(f'./data/ranked/{filename}', orient='records', lines=True)