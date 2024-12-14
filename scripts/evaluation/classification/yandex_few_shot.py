import pandas as pd
from sklearn.metrics import classification_report

from model.inference.yandex import YandexFewShot
from scripts.evaluation.classification.utils import save_classification_report

test_data = pd.read_json("./data/labeled/test.jsonl", lines=True)
test_data = test_data.sample(15)
test_data["text"] = test_data["text"].apply(lambda x: " ".join(x.split()))

topic_examples = (
    test_data[["text", "topic"]]
    .sample(15)
    .apply(lambda x: (x["text"], x["topic"]), axis=1)
    .tolist()
)

model = YandexFewShot(
    folder_id="b1g2v1k5v5v5v5v5v5",
    api_key="b1g2v1k5v5v5v5v5v5",
    examples=topic_examples,
)

predictions = model.predict(test_data["text"].tolist(), topic_examples)
true_labels = test_data["topic"].tolist()

report = classification_report(true_labels, predictions, output_dict=True)
save_classification_report(report, "Yandex-FewShot")
