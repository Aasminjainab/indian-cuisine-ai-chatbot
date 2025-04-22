import torch

class Config:
    # Model parameters
    model_name = "bert-base-uncased"
    max_length = 384
    batch_size = 8
    learning_rate = 3e-5
    epochs = 3
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths
    train_data_path = "data/indian_dishes.json"
    model_save_path = "save/bert_food_chatbot.pt"
    
    # Training parameters
    train_test_split = 0.2
    warmup_steps = 0
    weight_decay = 0.01
    max_answer_length = 30