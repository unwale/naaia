import torch
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt

def multifeature_collate_fn(batch):
    text_input_ids = pad_sequence([torch.tensor(item['text_input_ids']) for item in batch], batch_first=True)
    text_attention_mask = pad_sequence([torch.tensor(item['text_attention_mask']) for item in batch], batch_first=True)

    title_input_ids = pad_sequence([torch.tensor(item['title_input_ids']) for item in batch], batch_first=True)
    title_attention_mask = pad_sequence([torch.tensor(item['title_attention_mask']) for item in batch], batch_first=True)

    keyword_input_ids = pad_sequence([torch.tensor(item['keyword_input_ids']) for item in batch], batch_first=True)
    keyword_attention_mask = pad_sequence([torch.tensor(item['keyword_attention_mask']) for item in batch], batch_first=True)

    ner_input_ids = pad_sequence([torch.tensor(item['ner_input_ids']) for item in batch], batch_first=True)
    ner_attention_mask = pad_sequence([torch.tensor(item['ner_attention_mask']) for item in batch], batch_first=True)

    labels = torch.tensor([item['labels'] for item in batch])

    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'title_input_ids': title_input_ids,
        'title_attention_mask': title_attention_mask,
        'keyword_input_ids': keyword_input_ids,
        'keyword_attention_mask': keyword_attention_mask,
        'ner_input_ids': ner_input_ids,
        'ner_attention_mask': ner_attention_mask,
        'labels': labels
    }

def text_collate_fn(batch):
    text_input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True)
    text_attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True)

    labels = torch.tensor([item['labels'] for item in batch])

    return {
        'input_ids': text_input_ids,
        'attention_mask': text_attention_mask,
        'labels': labels
    }

def plot_training_history(loss, val_loss):
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


from transformers import DataCollatorWithPadding
from typing import Any, Dict

class CustomDataCollatorWithLabels(DataCollatorWithPadding):
    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = super().__call__(features)
        if "labels" in features[0]:
            batch["labels"] = [f["labels"] for f in features]
        return batch