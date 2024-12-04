import pandas as pd
from inference.sgd import SGDClassifier
from io import StringIO
import sys

test = pd.read_json('./data/labeled/test.jsonl', lines=True)
texts = test['lemmatized_text'].tolist()

model = SGDClassifier('model/saved/sgd.joblib')

import time
strt = time.time()
std = sys.stdout
sys.stdout = StringIO()
predictions = model.predict(texts, [i for i in range(13)])
sys.stdout = std
print('Time:', time.time() - strt)