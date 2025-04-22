from config import Config
from model import FoodChatbot
from preprocess import IndianFoodDataset
from train import train
from transformers import BertTokenizerFast
from torch.utils.data import random_split
import torch
import os

def main():
    # Initialize config
    config = Config()
    
    # Create save directory if it doesn't exist
    os.makedirs('save', exist_ok=True)
    
    # Initialize tokenizer with fast version
    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    
    # Load and split dataset
    print("Loading dataset...")
    full_dataset = IndianFoodDataset(config.train_data_path, tokenizer, config.max_length)
    
    # Calculate split sizes
    train_size = int((1 - config.train_test_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Initialize model
    print("Initializing model...")
    model = FoodChatbot(config)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Train model
    print("Starting training...")
    try:
        train(model, train_dataset, val_dataset, config)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 