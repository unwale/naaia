import pandas as pd
from sklearn.metrics import classification_report

from model.inference.sgd import SGDClassifier
from scripts import topics
from scripts.evaluation.classification.utils import save_classification_report

test = pd.read_json("./data/labeled/test.jsonl", lines=True)
texts = test["lemmatized_text"].tolist()

model = SGDClassifier("model/saved/sgd.joblib")

predictions = model.predict(texts, topics=topics)
true_labels = test["topic"].tolist()

report = classification_report(true_labels, predictions, output_dict=True)
save_classification_report(report, "sgd")
