import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

train = pd.read_json("./data/labeled/train.jsonl", lines=True)
test = pd.read_json("./data/labeled/test.jsonl", lines=True)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train["lemmatized_text"])
y_train = train["topic"]

model = SGDClassifier()
model.fit(X_train, y_train)

pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

joblib.dump(pipeline, "./model/saved/sgd.joblib")
