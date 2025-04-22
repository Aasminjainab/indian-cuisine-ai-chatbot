from transformers import BertTokenizerFast
import torch
import json

class IndianFoodChatbot:
    def __init__(self, model, config):
        self.model = model
        self.tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
        self.config = config
        
        # Load dataset for context
        with open(config.train_data_path, 'r') as f:
            self.data = json.load(f)['data']
    
    def find_relevant_context(self, query):
        """Find the most relevant context for the query"""
        query_words = set(query.lower().split())
        best_context = None
        max_matches = 0
        
        for dish in self.data:
            context = dish['paragraphs'][0]['context'].lower()
            matches = len(query_words.intersection(context.split()))
            if matches > max_matches:
                max_matches = matches
                best_context = dish['paragraphs'][0]['context']
        
        return best_context or self.data[0]['paragraphs'][0]['context']
    
    def get_response(self, query):
        """Generate response for user query"""
        # Find relevant context
        context = self.find_relevant_context(query)
        
        # Tokenize input
        inputs = self.tokenizer(
            query,
            context,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the most likely answer span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Ensure valid answer span
            if end_idx < start_idx:
                end_idx = start_idx + min(self.config.max_answer_length, 
                                        len(inputs['input_ids'][0]) - start_idx)
            
            # Extract answer tokens
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean up answer
            answer = answer.strip()
            if not answer:
                answer = "I'm sorry, I couldn't find a specific answer to that question."
        
        return answer