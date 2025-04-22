import gradio as gr
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import json
import os

class Config:
    model_name = "bert-base-uncased"
    max_length = 384
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_path = "data/indian_dishes.json"
    model_save_path = "save/bert_food_chatbot.pt"
    max_answer_length = 30

class FoodChatbot(torch.nn.Module):
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

class IndianFoodChatbotUI:
    def __init__(self):
        try:
            self.config = Config()
            
            print("Loading model...")
            self.model = FoodChatbot(self.config)
            
            # Load the trained model weights
            if os.path.exists(self.config.model_save_path):
                self.model.load_state_dict(torch.load(self.config.model_save_path, weights_only=True))
                print("Model loaded successfully!")
            else:
                raise FileNotFoundError(f"Model file not found at: {self.config.model_save_path}")
            
            self.model.to(self.config.device)
            self.model.eval()
            
            print("Loading tokenizer...")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name)
            
            # Load dataset for context
            print("Loading dataset...")
            if os.path.exists(self.config.train_data_path):
                with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)['data']
                print("Dataset loaded successfully!")
            else:
                print("Warning: Dataset file not found!")
                self.data = []
                
        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            raise e

    def find_relevant_context(self, query):
        if not self.data:
            return "No context available."
            
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
        try:
            context = self.find_relevant_context(query)
            
            inputs = self.tokenizer(
                query,
                context,
                max_length=self.config.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inputs['input_ids'][0][start_idx:end_idx + 1]
                )
            )
            
            return answer if answer else "I'm sorry, I couldn't find a specific answer to that question."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    def chatbot_response(self, message):
        if not message:
            return "Please ask a question about Indian cuisine."
        
        try:
            response = self.get_response(message)
            return response
        except Exception as e:
            print(f"Error in chatbot response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def create_gradio_interface():
    try:
        print("Initializing chatbot...")
        chatbot = IndianFoodChatbotUI()
        
        demo = gr.Interface(
            fn=chatbot.chatbot_response,
            inputs=[
                gr.Textbox(
                    lines=2,
                    placeholder="Ask me anything about Indian cuisine...",
                    label="Your Question"
                )
            ],
            outputs=[
                gr.Textbox(label="Assistant's Response")
            ],
            title="🍛 Indian Cuisine Assistant",
            description="""
            Welcome to the Indian Cuisine Assistant! I can help you with:
            - Recipe ingredients and cooking times
            - Regional origins of dishes
            - Dietary information
            - Cooking instructions
            """,
            examples=[
                ["What are the ingredients in Butter Chicken?"],
                ["How long does it take to cook Biryani?"],
                ["Is Dosa vegetarian?"],
                ["Which region is Sambar from?"]
            ],
            theme=gr.themes.Soft()
        )
        
        return demo
        
    except Exception as e:
        print(f"Error creating Gradio interface: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        print("Starting Gradio server...")
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        print(f"Error launching Gradio server: {str(e)}")