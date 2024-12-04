import os
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import AIMessage
from langchain_gigachat.chat_models import GigaChat

load_dotenv()


class GigaChatZeroshot():

    def __init__(self, model: str = 'GigaChat'):
        """
        Initializes the GigaChat Zeroshot classifier.
        """
        self.model = GigaChat(
            credentials=os.getenv('GIGACHAT_AUTH_TOKEN'),
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
        prompt = f'Ты - классификатор новостных постов. Текст: {' '.join(text.split())}\n\n Ты должен сопоставить тексту одну тему из этого списка: {', '.join(topics)}. В ответе должна быть только тема и только из этого списка. Тема: '
        
        response: AIMessage = self.model.invoke(prompt)
        response = response.content
        for topic in topics:
            if topic in response:
                return topic
        return 'Другое'