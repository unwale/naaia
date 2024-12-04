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
data.drop(columns=['text', 'title'], inplace=True)
data['queries'] = data['queries'].apply(json.loads)
assert len(data['queries'].values[0]) <= 3

batch_size = 3000
batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]

response_schema = {
    'name': 'TemporalClassificationResult',
    'schema': {
        'type': 'object',
        'properties': {
            'temporal_intents': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'temporal_intent': {
                            'type': 'string'
                        }
                    }
                }
            }
        }
    }
}

intents = ['recent', 'historical', 'neutral']

data['temporal_intent'] = None
for i, batch in enumerate(batches):
    print(f'Batch {i} started')
    prompts = []
    for queries in batch['queries'].values:
        prompt = f'Классифицируй каждый пользовательский запрос по темпоральному интенту (recent - ): {', '.join(intents)}.\n\n'
        for j, q in enumerate(queries):
            prompt += f'Запрос {j+1}: \"{q}\"\n\n'
        prompt += '\n\nОтвечай в формате JSON с массивом "temporal_intents"'
        prompts.append(prompt)
    prompts = pd.Series(prompts)
    batch_id = 'batch_674f17469cb88190b1da60054e586529' #make_batch(openai, args.model, prompts, response_schema)
    results = watch_batch(openai, batch_id, f'{args.output[:-6]}{i}.jsonl')
    temporal_intents = parse_batch_results(results)
    print(temporal_intents[:5])
    print(len(prompts), len(temporal_intents))
    temporal_intents = [json.loads(label)['temporal_intents'] for label in temporal_intents]
    temporal_intents = [ [intent['temporal_intent'] for intent in intents] for intents in temporal_intents]
    print(batch.index.shape, len(temporal_intents))
    data.loc[batch.index, 'temporal_intent'] = temporal_intents
    data.loc[batch.index, 'temporal_intent'] = data.loc[batch.index, 'temporal_intent'].apply(json.dumps)
    print(f'Batch {i} done')

data.to_json(args.output, orient='records', lines=True)