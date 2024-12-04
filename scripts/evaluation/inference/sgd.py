import joblib
from typing import List

class SGDClassifier():

    def __init__(self, path):
        self.model = joblib.load(path)

    def predict(self, inputs: List[str], topics: List[str]) -> List[str]:
        return self.model.predict(inputs)
