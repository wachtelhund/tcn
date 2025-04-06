import torch
import torch.nn as nn
import torch.optim as optim
from tcn import TemporalConvNet
from data_loader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG

def train_model(model, train_loader, test_loader, num_epochs=None, learning_rate=None, patience=None):
    # Use config values if not specified
    num_epochs = num_epochs or TRAINING_CONFIG['num_epochs']
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    patience = patience or TRAINING_CONFIG['patience']
    checkpoint_dir = TRAINING_CONFIG['checkpoint_dir']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nTraining model for {num_epochs} epochs with learning rate {learning_rate}")
    print(f"Early stopping patience: {patience}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.transpose(1, 2))  # Transpose to match TCN input shape
            loss = criterion(outputs[:, -1], batch_y)  # Only use last prediction
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x.transpose(1, 2))
                loss = criterion(outputs[:, -1], batch_y)
                test_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        
        # Update learning rate based on test loss
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(checkpoint_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}: Saved new best model with test loss: {test_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def plot_predictions(model, test_loader, scalers, target_features):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.transpose(1, 2))
            predictions.append(outputs[:, -1].numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # Denormalize the data
    for i, feature in enumerate(target_features):
        predictions[:, i] = predictions[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
        actuals[:, i] = actuals[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
    
    # Create plots directory
    os.makedirs(VIZ_CONFIG['save_dir'], exist_ok=True)
    
    # Plot each target feature
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(target_features):
        plt.subplot(1, len(target_features), i+1)
        plt.plot(actuals[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(feature)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_CONFIG['save_dir'], 'predictions.png'))
    plt.close()
    
    print(f"Saved prediction plots to {VIZ_CONFIG['save_dir']}/predictions.png")

def main():
    # Get configuration
    batch_size = MODEL_CONFIG['batch_size']
    num_channels = MODEL_CONFIG['num_channels']
    kernel_size = MODEL_CONFIG['kernel_size']
    dropout = MODEL_CONFIG['dropout']
    
    sequence_length = DATA_CONFIG['sequence_length']
    target_features = DATA_CONFIG['target_features']
    input_features = DATA_CONFIG['input_features']
    
    # Get data loaders
    train_loader, test_loader, scalers = get_data_loaders(
        batch_size=batch_size,
        sequence_length=sequence_length,
        target_features=target_features,
        input_features=input_features
    )
    
    # Get number of input and output features
    num_inputs = len(train_loader.dataset.input_features)
    num_outputs = len(target_features)
    
    print(f"\nInitializing TCN model:")
    print(f"  Input features: {num_inputs}")
    print(f"  Output features: {num_outputs}")
    print(f"  Channels: {num_channels}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Dropout: {dropout}")
    
    # Initialize model
    model = TemporalConvNet(
        num_inputs=num_inputs, 
        num_outputs=num_outputs, 
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    
    # Train model
    train_losses, test_losses = train_model(model, train_loader, test_loader)
    
    # Plot predictions
    if VIZ_CONFIG['plot_predictions']:
        plot_predictions(model, test_loader, scalers, target_features)
    
    # Plot losses
    if VIZ_CONFIG['plot_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(VIZ_CONFIG['save_dir'], 'losses.png'))
        plt.close()
        print(f"Saved loss plots to {VIZ_CONFIG['save_dir']}/losses.png")

if __name__ == '__main__':
    main() 