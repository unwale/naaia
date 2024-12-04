import torch
import torch.nn as nn
from transformers import AutoModel

class RuBERTLinText(nn.Module):
    
    def __init__(self, model: str,  num_classes: int):
        super(RuBERTLinText, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        logits = self.linear(cls_token)
        return logits

class RuBERTMLPText(nn.Module):
    
    def __init__(self, model: str, num_classes: int, hidden_size: int):
        super(RuBERTMLPText, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        logits = self.mlp(cls_token)
        return logits

class RuBERTMultifeature(nn.Module):
    
    def __init__(self, model: str, num_classes: int, hidden_size: int):
        super(RuBERTMultifeature, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.title_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.keywords_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.ner_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, 
                text_input_ids, text_attention_mask,
                title_input_ids, title_attention_mask,
                keywords_input_ids, keywords_attention_mask,
                ner_input_ids, ner_attention_mask):
        
        text_outputs = self.bert(text_input_ids, text_attention_mask)
        text_last_hidden_state = text_outputs.last_hidden_state
        text_cls_token = text_last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_cls_token)

        title_outputs = self.bert(title_input_ids, title_attention_mask)
        title_last_hidden_state = title_outputs.last_hidden_state
        title_cls_token = title_last_hidden_state[:, 0, :]
        title_proj = self.title_proj(title_cls_token)

        keywords_outputs = self.bert(keywords_input_ids, keywords_attention_mask)
        keywords_last_hidden_state = keywords_outputs.last_hidden_state
        keywords_cls_token = keywords_last_hidden_state[:, 0, :]
        keywords_proj = self.keywords_proj(keywords_cls_token)

        ner_outputs = self.bert(ner_input_ids, ner_attention_mask)
        ner_last_hidden_state = ner_outputs.last_hidden_state
        ner_cls_token = ner_last_hidden_state[:, 0, :]
        ner_proj = self.ner_proj(ner_cls_token)

        features = torch.cat([text_proj, title_proj, keywords_proj, ner_proj], dim=1)
        logits = self.mlp(features)

        return logits

class RuBERTTextTags(nn.Module):
    
    def __init__(self, model: str, num_classes: int, hidden_size: int):
        super(RuBERTMultifeature, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.tags_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, 
                text_input_ids, text_attention_mask,
                tags_input_ids, tags_attention_mask):
        
        text_outputs = self.bert(text_input_ids, text_attention_mask)
        text_last_hidden_state = text_outputs.last_hidden_state
        text_cls_token = text_last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_cls_token)

        tags_outputs = self.bert(tags_input_ids, tags_attention_mask)
        tags_last_hidden_state = tags_outputs.last_hidden_state
        tags_cls_token = tags_last_hidden_state[:, 0, :]
        tags_proj = self.tags_proj(tags_cls_token)

        features = torch.cat([text_proj, tags_proj], dim=1)
        logits = self.mlp(features)

        return logits