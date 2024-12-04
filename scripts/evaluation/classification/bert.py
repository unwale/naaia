from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sklearn.metrics import classification_report
from scripts.evaluation.classification.utils import save_classification_report
from scripts.evaluation.inference.bert import BertTextClassifier
from scripts.training.topic_classification.models import *
import argparse

parser = argparse.ArgumentParser(description='Evaluate a linear model on top of RuBERT')
parser.add_argument('--model', type=str, help='Model path')
parser.add_argument('--tokenizer', type=str, help='Tokenizer path') 
# TODO add tokenizer.model_max_length for deeppavlov
args = parser.parse_args()

model_name = args.model.split('/')[-1].split('.')[0]
model = BertTextClassifier(args.model, args.tokenizer)

test = pd.read_json('./data/labeled/test.jsonl', lines=True)
train = pd.read_json('./data/labeled/train.jsonl', lines=True)

texts = test['text'].tolist()   
topics = train['topic'].unique().tolist() # TODO figure out the correct order of topics OR retrain all models

gt = test['topic'].tolist()

predictions = model.predict(texts, topics)
report = classification_report(gt, predictions, output_dict=True)
save_classification_report(report, model_name)