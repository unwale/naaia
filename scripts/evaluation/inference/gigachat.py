import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_gigachat.chat_models import GigaChat

load_dotenv()


class GigaChatZeroshot:

    def __init__(self, model: str = "GigaChat"):
        """
        Initializes the GigaChat Zeroshot classifier.
        """
        self.model = GigaChat(
            credentials=os.getenv("GIGACHAT_AUTH_TOKEN"),
            model=model,
            verify_ssl_certs=False,
        )

    def predict(self, inputs: List[str], topics: List[str]) -> List[str]:
        predictions = []
        for text in inputs:
            prediction = self._classify_sample_zero_shot(text, topics)
            predictions.append(prediction)

        return predictions

    def _classify_sample_zero_shot(self, text: str, topics: List[str]) -> str:
        text = " ".join(text.split())
        topics_text = ", ".join(topics)
        prompt = (
            "Ты - классификатор новостных постов. "
            + f"Текст: {text}\n\n Ты должен сопоставить тексту"
            + f" одну тему из этого списка: {topics_text}. В ответе должна"
            + " быть только тема и только из этого списка. Тема: "
        )

        response: AIMessage = self.model.invoke(prompt)
        response = response.content
        for topic in topics:
            if topic in response:
                return topic
        return "Другое"
