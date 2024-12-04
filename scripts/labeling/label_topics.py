import os
import argparse
import json 
import pandas as pd
from openai import OpenAI
from utils import make_batch, watch_batch, parse_batch_results
from dotenv import load_dotenv
load_dotenv()

argparser = argparse.ArgumentParser(description='Label topics using OpenAI Batch API')
argparser.add_argument('--model', type=str, help='Model name', default='gpt-4o-mini')
argparser.add_argument('--input', type=str, help='Path to jsonl data to label')
argparser.add_argument('--output', type=str, help='Path to save the output')
args = argparser.parse_args()

openai = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

data = pd.read_json(args.input, lines=True)

batch_size = 3000
batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]

response_schema = {
    'name': 'TopicClassificationResult',
    'schema': {
        'type': 'object',
        'properties': {
            'topic': {
                'type': 'string'
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

data['topic'] = None
for i, batch in enumerate(batches):
    print(f'Batch {i} started')
    prompts = []
    for text in batch['text'].values:
        prompt = f'Классифицируй следующий текст по одной из этих тем: {', '.join(topics)}.\n\nТекст: \"{text}\"\n\nОтвечай в формате JSON с полем "topic"'
        prompts.append(prompt)
    prompts = pd.Series(prompts)
    batch_id = make_batch(openai, args.model, prompts, response_schema)
    results = watch_batch(openai, batch_id, f'{args.output[:-6]}{i}.jsonl')
    topic_labels = parse_batch_results(results).values
    print(topic_labels[:5])
    topic_labels = [json.loads(label[0])['topic'] for label in topic_labels]
    data.loc[batch.index, 'topic'] = topic_labels
    print(f'Batch {i} done')

data.to_json(args.output, orient='records', lines=True)