import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from scripts import topics

threshold = 80

train = pd.read_json("./data/ranked/test.jsonl", lines=True)
test = pd.read_json("./data/ranked/test.jsonl", lines=True)
train["topic_ranking"] = train["topic_ranking"].apply(json.loads)
test["topic_ranking"] = test["topic_ranking"].apply(json.loads)
train["topic_list"] = train["topic_ranking"].apply(
    lambda x: [t for t in topics if x[t] >= threshold]
)
test["topic_list"] = test["topic_ranking"].apply(
    lambda x: [t for t in topics if x[t] >= threshold]
)

mlb = MultiLabelBinarizer(classes=topics)
y_train = mlb.fit_transform(train["topic_list"])
y_test = mlb.transform(test["topic_list"])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train["lemmatized_text"])
X_test = vectorizer.transform(test["lemmatized_text"])

base_model = SGDClassifier()
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

joblib.dump(pipeline, "./model/saved/sgd_multilabel.joblib")

joblib.dump(mlb, "./model/saved/mlb.joblib")
