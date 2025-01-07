import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from model.inference.yandex import YandexFewShot
from scripts import topics
from scripts.evaluation.classification.utils import save_classification_report

load_dotenv(".../.env")

test_data = pd.read_json("./data/labeled/test.jsonl", lines=True).sample(150)
test_data["text"] = test_data["text"].apply(lambda x: " ".join(x.split()))

topic_examples = (
    test_data[["text", "topic"]]
    .sample(15)
    .apply(lambda x: (x["text"], x["topic"]), axis=1)
    .tolist()
)

model = YandexFewShot(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    api_key=os.getenv("YANDEX_IAM"),
    examples=topic_examples,
)

predictions = model.predict(test_data["text"].tolist(), topics)
true_labels = test_data["topic"].tolist()

report = classification_report(true_labels, predictions, output_dict=True)
save_classification_report(report, "Yandex-FewShot")
