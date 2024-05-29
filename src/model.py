import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class BERTModel(nn.Module):
    def __init__(self, model_checkpoint='bert') -> None:
        super().__init__()

        self.model_checkpoint = model_checkpoint
        if model_checkpoint == 'roberta':
            self.base_model = RobertaModel.from_pretrained('FacebookAI/roberta-base')
        elif model_checkpoint == 'bert':
            self.base_model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.1)
        self.classification_head = nn.Linear(self.base_model.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.dropout(x)
        x = self.classification_head(x)
        return x
    

