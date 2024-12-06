import pandas as pd

from model.inference.sgd import SGDClassifier

test = pd.read_json("./data/labeled/test.jsonl", lines=True)
texts = test["lemmatized_text"].tolist()

model = SGDClassifier("model/saved/sgd.joblib")

predictions = model.predict(texts, [i for i in range(13)])

# TODO make and save a report
