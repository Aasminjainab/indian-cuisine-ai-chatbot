from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def train(model, train_dataset, val_dataset, config):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size
    )
    
    # Use PyTorch's AdamW instead of transformers' version
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    model.to(config.device)
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(config.device) for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Saved new best model with validation loss: {avg_val_loss}")