import torch.nn as nn
from transformers import AutoModel


class RuBERTMLPText(nn.Module):

    def __init__(self, model: str, num_classes: int, hidden_size: int):
        super(RuBERTMLPText, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        logits = self.mlp(cls_token)
        return logits
