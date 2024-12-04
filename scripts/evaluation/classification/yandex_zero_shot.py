import pandas as pd
from inference.yandex import YandexZeroshot
from sklearn.metrics import classification_report
from utils import save_classification_report

model = YandexZeroshot(folder_id='b1g2v1k5v5v5v5v5v5', api_key='b1g2v1k5v5v5v5v5v5')

test_data = pd.read_json('./data/labeled/test.jsonl', lines=True)
test_data = test_data.sample(15)
test_data['text'] = test_data['text'].apply(lambda x: ' '.join(x.split()))

predictions = model.predict(test_data['text'].tolist(), topics=['sport', 'politics', 'economics', 'culture', 'science'])
true_labels = test_data['topic'].tolist()

report = classification_report(true_labels, predictions, output_dict=True)
save_classification_report(report, 'Yandex-Zeroshot')