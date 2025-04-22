from config import Config
from model import FoodChatbot
from chatbot import IndianFoodChatbot
import torch

def main():
    try:
        # Initialize config
        config = Config()
        
        # Initialize model
        print("Loading model...")
        model = FoodChatbot(config)
        
        # Load trained weights
        model.load_state_dict(torch.load(config.model_save_path))
        model.to(config.device)
        
        # Create chatbot interface
        chatbot = IndianFoodChatbot(model, config)
        
        print("\nIndian Food Chatbot Ready!")
        print("You can ask questions about Indian dishes, their ingredients, cooking times, and regional origins.")
        print("Type 'quit' to exit.\n")
        
        # Interactive loop
        while True:
            query = input("You: ")
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            try:
                response = chatbot.get_response(query)
                print(f"Bot: {response}\n")
            except Exception as e:
                print(f"Error: Could not generate response. {str(e)}\n")
                
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")

if __name__ == "__main__":
    main() 