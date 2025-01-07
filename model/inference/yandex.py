from typing import List, Tuple

from tqdm import tqdm
from yandex_cloud_ml_sdk import YCloudML


class YandexZeroshot:

    def __init__(self, folder_id: str, api_key: str):
        """
        Initializes the Yandex Zeroshot classifier.

        Parameters:
        - folder_id: The Yandex folder ID.
        - api_key: The Yandex API key.
        """
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)

    def predict(self, inputs: List[str], topics: List[str]) -> List[str]:
        model = self.sdk.models.text_classifiers("yandexgpt").configure(
            task_description="Определи тему новости", labels=topics
        )
        predictions = []
        for text in tqdm(inputs, total=len(inputs)):
            prediction = max(
                model.run(text).predictions, key=lambda x: x.confidence
            ).label
            predictions.append(prediction)

        return predictions


class YandexFewShot:

    def __init__(
        self, folder_id: str, api_key: str, examples: List[Tuple[str, str]]
    ):
        """
        Initializes the Yandex Fewshot classifier.

        Parameters:
        - folder_id: The Yandex folder ID.
        - api_key: The Yandex API key.
        - examples: A list of tuples, where each tuple contains
                    a text and a label.
        """
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        self.examples = examples

    def predict(self, inputs: List[str], topics: List[str]) -> List[str]:
        model = self.sdk.models.text_classifiers("yandexgpt").configure(
            task_description="Определи тему новости",
            labels=topics,
            samples=self.examples,
        )

        predicitions = []
        for text in tqdm(inputs, total=len(inputs)):
            prediction = max(
                model.run(text).predictions, key=lambda x: x.confidence
            ).label
            predicitions.append(prediction)

        return predicitions
