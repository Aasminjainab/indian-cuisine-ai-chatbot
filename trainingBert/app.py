import gradio as gr
from config import Config
from model import FoodChatbot
from transformers import BertTokenizerFast
import torch
import json
import os

class IndianFoodChatbotUI:
    def __init__(self):
        try:
            self.config = Config()
            
            # Ensure model directory exists
            os.makedirs('save', exist_ok=True)
            
            print("Loading model...")
            self.model = FoodChatbot(self.config)
            
            # Check if model file exists
            if os.path.exists(self.config.model_save_path):
                self.model.load_state_dict(torch.load(self.config.model_save_path, weights_only=True))
                print("Model loaded successfully!")
            else:
                print("Warning: Model file not found. Please train the model first.")
            
            self.model.to(self.config.device)
            self.model.eval()
            
            print("Loading tokenizer...")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name)
            
            # Load dataset
            print("Loading dataset...")
            if os.path.exists(self.config.train_data_path):
                with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)['data']
                print("Dataset loaded successfully!")
            else:
                print("Error: Dataset file not found!")
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
                return_tensors='pt',
                return_token_type_ids=True
            )
            
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                start_idx = torch.argmax(start_scores)
                end_idx = torch.argmax(end_scores)
                
                if end_idx < start_idx:
                    end_idx = start_idx + min(self.config.max_answer_length, 
                                            len(inputs['input_ids'][0]) - start_idx)
                
                answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                answer = answer.strip()
                if not answer:
                    answer = "I'm sorry, I couldn't find a specific answer to that question."
            
            return answer
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    def chatbot_response(self, message, history):
        if not message:
            return "Please ask a question about Indian cuisine."
        
        try:
            response = self.get_response(message)
            return response
        except Exception as e:
            print(f"Error in chatbot response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def create_demo():
    try:
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = IndianFoodChatbotUI()
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .bot-message {
            background-color: #f5f5f5;
        }
        """
        
        # Create the Gradio interface
        demo = gr.Interface(
            fn=chatbot.chatbot_response,
            inputs=gr.Textbox(
                lines=2,
                placeholder="Ask me anything about Indian cuisine...",
                label="Your Question"
            ),
            outputs=gr.Textbox(label="Assistant's Response"),
            title="🍛 Indian Cuisine Assistant",
            description="""
            Welcome to the Indian Cuisine Assistant! I can help you with:
            - Recipe ingredients and cooking times
            - Regional origins of dishes
            - Dietary information (vegetarian/non-vegetarian)
            - Cooking instructions and preparation times
            """,
            examples=[
                ["What are the ingredients in Butter Chicken?"],
                ["How long does it take to cook Biryani?"],
                ["Is Dosa vegetarian?"],
                ["Which region is Sambar from?"],
                ["What is the cooking time for Tandoori Chicken?"]
            ],
            css=custom_css,
            theme=gr.themes.Soft()
        )
        
        return demo
        
    except Exception as e:
        print(f"Error creating Gradio demo: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        print("Starting Gradio server...")
        demo = create_demo()
        # Start the server with specific port and host
        demo.launch(
            server_name="0.0.0.0",  # Allows external connections
            server_port=7860,        # Specific port
            share=True,              # Creates public URL
            debug=True              # Shows detailed errors
        )
    except Exception as e:
        print(f"Error launching Gradio server: {str(e)}") 