from typing import List

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from scripts.training.topic_classification.models import (
    RuBERTLinText,
    RuBERTMLPText,
    RuBERTMultifeature,
)


class BertTextClassifier:

    def __init__(
        self, model_path: str, base_model_name: str, tokenizer_path: str
    ):
        self.model = self._make_model_based_on_name(
            model_path, base_model_name
        )
        self.model.load_state_dict(torch.load(model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.model_max_length = 512

    def predict(
        self, inputs: List[str], topics: List[str], batch_size=8
    ) -> List[str]:
        predictions = []
        tokenized_inputs = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_inputs = {
            key: value.to(self.device)
            for key, value in tokenized_inputs.items()
        }
        tokenized_inputs.pop("token_type_ids")
        for i in tqdm(
            range(0, len(inputs), batch_size),
            desc="Predicting topics",
            total=len(inputs) // batch_size,
        ):
            batch = {
                key: value[i : i + batch_size]
                for key, value in tokenized_inputs.items()
            }

            with torch.no_grad():
                logits = self.model(**batch)
                predictions.extend(torch.argmax(logits, dim=1))

        return [topics[prediction] for prediction in predictions]

    def _make_model_based_on_name(self, model_path: str, base_model_name: str):
        model_suffix = model_path.split("-")[-1].replace(".pth", "")
        match model_suffix:
            case "lin":
                return RuBERTLinText(base_model_name)
            case "mlp":
                return RuBERTMLPText(base_model_name)
            case "multifeature":
                return RuBERTMultifeature(base_model_name)
            case _:
                raise ValueError(f"Unknown model suffix: {model_suffix}")
