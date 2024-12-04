import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
# from scripts.evaluation.utils import save_classification_report

train = pd.read_json('./data/labeled/train.jsonl', lines=True)
test = pd.read_json('./data/labeled/test.jsonl', lines=True)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['lemmatized_text'])
y_train = train['topic']

model = SGDClassifier()
model.fit(X_train, y_train)


# build and save pipeline

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('model', model)
])

joblib.dump(pipeline, './model/saved/sgd.joblib')