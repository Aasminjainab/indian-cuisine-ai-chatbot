from transformers import BertTokenizerFast
import json
import torch
from torch.utils.data import Dataset

class IndianFoodDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)['data']
            
        self.examples = []
        
        # Process data
        for dish in self.data:
            for paragraph in dish['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer_text = qa['answers'][0]['text']
                    answer_start = context.find(answer_text)
                    
                    # Skip examples where answer can't be found in context
                    if answer_start == -1:
                        continue
                        
                    self.examples.append({
                        'question': question,
                        'context': context,
                        'answer_text': answer_text,
                        'answer_start': answer_start,
                        'answer_end': answer_start + len(answer_text)
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
            return_token_type_ids=True
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Find start and end token positions for the answer
        offset_mapping = encoding.pop('offset_mapping').tolist()
        token_type_ids = encoding['token_type_ids'].tolist()
        
        start_position = 0
        end_position = 0
        
        # Find the token positions that correspond to the answer
        for idx, (token_type_id, offset) in enumerate(zip(token_type_ids, offset_mapping)):
            if token_type_id == 1:  # Only look in context tokens, not question tokens
                if offset[0] <= example['answer_start'] <= offset[1]:
                    start_position = idx
                if offset[0] <= example['answer_end'] <= offset[1]:
                    end_position = idx
                    break
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'start_positions': torch.tensor(start_position),
            'end_positions': torch.tensor(end_position)
        } 