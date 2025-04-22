from config import Config
from model import FoodChatbot
from transformers import BertTokenizerFast
import torch
import json

def load_test_questions():
    test_questions = [
        "What are the main ingredients in Butter Chicken?",
        "How long does it take to cook Biryani?",
        "Is Dosa vegetarian?",
        "Which region is Sambar from?",
        "What is the cooking time for Tandoori Chicken?"
    ]
    return test_questions

def test_model(model, tokenizer, config):
    model.eval()
    test_questions = load_test_questions()
    
    # Load context from dataset
    with open(config.train_data_path, 'r') as f:
        data = json.load(f)['data']
    
    for question in test_questions:
        # Find relevant context
        context = ""
        for dish in data:
            if any(keyword in dish['title'].lower() for keyword in question.lower().split()):
                context = dish['paragraphs'][0]['context']
                break
        
        if not context:
            context = data[0]['paragraphs'][0]['context']  # Default context
        
        # Tokenize
        inputs = tokenizer(
            question,
            context,
            max_length=config.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=True  # Explicitly request token_type_ids
        )
        
        # Move to device
        inputs = {k: v.to(config.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Get the most likely answer span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Ensure end_idx is not before start_idx
            if end_idx < start_idx:
                end_idx = start_idx + min(config.max_answer_length, 
                                        len(inputs['input_ids'][0]) - start_idx)
            
            # Get answer tokens and decode
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean up answer
            answer = answer.strip()
            if not answer:
                answer = "Could not find an answer"
            
        print(f"Q: {question}")
        print(f"A: {answer}\n")

def main():
    config = Config()
    
    # Load model
    print("Loading model...")
    model = FoodChatbot(config)
    model.load_state_dict(torch.load(config.model_save_path, weights_only=True))  # Added weights_only=True
    model.to(config.device)
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    
    print("Running tests...")
    test_model(model, tokenizer, config)

if __name__ == "__main__":
    main()