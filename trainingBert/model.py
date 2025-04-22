from transformers import BertForQuestionAnswering
import torch.nn as nn

class FoodChatbot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertForQuestionAnswering.from_pretrained(
            config.model_name,
            return_dict=True
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        return outputs