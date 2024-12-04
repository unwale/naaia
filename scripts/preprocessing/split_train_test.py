import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('split_train_test')

parser = argparse.ArgumentParser(description='Split a jsonl file into train and test sets')
parser.add_argument('--input', type=str, help='Path to input jsonl file', default='.../data/raw/data.jsonl')
parser.add_argument('--output-train', type=str, help='Path to save the training set', default='.../data/raw/train.jsonl')
parser.add_argument('--output-test', type=str, help='Path to save the test set', default='.../data/raw/test.jsonl')
parser.add_argument('--n', type=int, help='Number of samples to take', default=10000)
parser.add_argument('--ratio', type=float, help='Train-test split ratio', default=0.9)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
args = parser.parse_args()

logger.info('Reading data')
data = pd.read_json(args.input, lines=True)
data = data[data['text'].str.len() > 50]
data = data.dropna(subset=['text'])
data = data.sample(args.n, random_state=args.seed)

logger.info(f'Splitting data into train and test sets with ratio {args.ratio}')
train = data.sample(frac=args.ratio, random_state=args.seed)
test = data.drop(train.index)

train.to_json(args.output_train, orient='records', lines=True)
test.to_json(args.output_test, orient='records', lines=True)
logger.info(f'Data saved to {args.output_train} and {args.output_test}')