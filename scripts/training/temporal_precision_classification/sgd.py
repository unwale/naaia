import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

data = pd.read_json("./data/time_precision/data.jsonl", lines=True)

train, test = train_test_split(data, test_size=0.2, random_state=42)

model = make_pipeline(
    TfidfVectorizer(max_features=3000),
    SGDClassifier(class_weight="balanced", random_state=42),
)
model.fit(train["context"], train["precision"])

joblib.dump(model, "./model/saved/time_precision_model.joblib")
