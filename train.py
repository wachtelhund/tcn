import torch
import torch.nn as nn
import torch.optim as optim
from tcn import TemporalConvNet
from data_loader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import os
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
import pandas as pd

def train_model(model, train_loader, test_loader, num_epochs=None, learning_rate=None, patience=None, weight_decay=None):
    # Use config values if not specified
    num_epochs = num_epochs or TRAINING_CONFIG['num_epochs']
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    patience = patience or TRAINING_CONFIG['patience']
    weight_decay = weight_decay or TRAINING_CONFIG.get('weight_decay', 0)
    checkpoint_dir = TRAINING_CONFIG['checkpoint_dir']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    lr_scheduler_type = TRAINING_CONFIG.get('lr_scheduler', 'reduce_plateau')
    reduce_factor = TRAINING_CONFIG.get('reduce_factor', 0.5)
    reduce_patience = TRAINING_CONFIG.get('reduce_patience', 5)
    
    if lr_scheduler_type == 'reduce_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=reduce_factor, 
            patience=reduce_patience, verbose=True
        )
    elif lr_scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate/10
        )
    elif lr_scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=reduce_patience, gamma=reduce_factor
        )
    else:
        # Default to ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=reduce_factor, 
            patience=reduce_patience, verbose=True
        )
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nTraining model for {num_epochs} epochs with learning rate {learning_rate}")
    print(f"Early stopping patience: {patience}")
    if weight_decay > 0:
        print(f"Using L2 regularization with weight decay: {weight_decay}")
    
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
        
        # Update learning rate based on scheduler type
        if lr_scheduler_type == 'reduce_plateau':
            scheduler.step(test_loss)
        else:
            scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(checkpoint_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, model_path)
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
    dates = []
    
    # Check if we have date column in the dataset
    date_column = DATA_CONFIG['date_column']
    has_dates = hasattr(test_loader.dataset, 'data') and date_column in test_loader.dataset.data.columns
    is_hourly = DATA_CONFIG.get('is_hourly', False)
    forecast_horizon = DATA_CONFIG.get('forecast_horizon', 24 if is_hourly else 1)
    
    with torch.no_grad():
        # Get all predictions from test loader
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.transpose(1, 2))
            predictions.append(outputs[:, -1].numpy())
            actuals.append(batch_y.numpy())
    
    # For normal prediction evaluation
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # Get dates for x-axis if available
    if has_dates:
        # Get dates from the test dataset, skipping sequence_length entries at the beginning
        dates = test_loader.dataset.data[date_column].iloc[test_loader.dataset.sequence_length:].reset_index(drop=True)
        # Make sure we have the right number of dates
        if len(dates) > len(actuals):
            dates = dates[:len(actuals)]
    
    # Denormalize the data
    for i, feature in enumerate(target_features):
        predictions[:, i] = predictions[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
        actuals[:, i] = actuals[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
    
    # Create plots directory
    os.makedirs(VIZ_CONFIG['save_dir'], exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a separate plot for each target feature
    for i, feature in enumerate(target_features):
        # Calculate MSE and RMSE for the limited forecast horizon, not all predictions
        # Use only as many predictions as specified by forecast_horizon
        if len(actuals) > forecast_horizon:
            forecast_mse = np.mean((predictions[:forecast_horizon, i] - actuals[:forecast_horizon, i])**2)
            forecast_rmse = np.sqrt(forecast_mse)
        else:
            forecast_mse = np.mean((predictions[:, i] - actuals[:, i])**2)
            forecast_rmse = np.sqrt(forecast_mse)
        
        # Calculate metrics for all predictions (for comparison)
        full_mse = np.mean((predictions[:, i] - actuals[:, i])**2)
        full_rmse = np.sqrt(full_mse)
        
        # Calculate percentage error metrics
        mape = np.mean(np.abs((actuals[:, i] - predictions[:, i]) / np.maximum(1e-5, np.abs(actuals[:, i])))) * 100
        
        # Print metrics
        print(f"\nMetrics for {feature}:")
        print(f"  {forecast_horizon}-step Forecast MSE: {forecast_mse:.4f}")
        print(f"  {forecast_horizon}-step Forecast RMSE: {forecast_rmse:.4f}")
        print(f"  Full Test Set MSE: {full_mse:.4f}")
        print(f"  Full Test Set RMSE: {full_rmse:.4f}")
        print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Create three different plots
        
        # 1. Short-term forecast plot (immediate forecast horizon)
        plt.figure(figsize=(12, 6))
        
        if has_dates and len(dates) == len(actuals):
            # Only show the forecast horizon in the plot
            limit = min(forecast_horizon, len(dates))
            limited_dates = dates[:limit]
            limited_actuals = actuals[:limit, i]
            limited_preds = predictions[:limit, i]
            
            unit = "hours" if is_hourly else "days"
            
            plt.plot(limited_dates, limited_actuals, label='Actual', color='blue')
            plt.plot(limited_dates, limited_preds, label=f'Predicted ({forecast_horizon} {unit})', color='red')
            
            # Format x-axis to show dates
            if is_hourly:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                # Set major ticks to show every few hours
                interval = max(1, limit // 10)
                plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            else:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                # Set number of ticks based on data length
                interval = max(1, limit // 10)
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            
            # Automatically rotate date labels for better readability
            fig = plt.gcf()
            fig.autofmt_xdate()
        else:
            # Fallback to index-based plotting if dates are not available
            limit = min(forecast_horizon, len(actuals))
            limited_actuals = actuals[:limit, i]
            limited_preds = predictions[:limit, i]
            x_values = range(len(limited_preds))
            
            plt.plot(x_values, limited_actuals, label='Actual')
            plt.plot(x_values, limited_preds, label=f'Predicted ({forecast_horizon} steps)', color='red')
            
            # Use integer ticks for x-axis
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        title_text = f'Short-term Forecast for {feature}'
        title_text += f' (RMSE: {forecast_rmse:.4f})'
        plt.title(title_text)
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        
        # Save with timestamp in filename
        filename = f"{feature}_short_term_{timestamp}.png"
        filepath = os.path.join(VIZ_CONFIG['save_dir'], filename)
        plt.savefig(filepath)
        plt.close()
        
        # 2. Medium-term forecast plot (about a week for hourly, a month for daily)
        plt.figure(figsize=(12, 6))
        
        medium_horizon = 168 if is_hourly else 30  # A week for hourly, a month for daily
        if has_dates and len(dates) == len(actuals):
            # Show a medium-term view
            limit = min(medium_horizon, len(dates))
            medium_dates = dates[:limit]
            medium_actuals = actuals[:limit, i]
            medium_preds = predictions[:limit, i]
            
            plt.plot(medium_dates, medium_actuals, label='Actual', color='blue')
            plt.plot(medium_dates, medium_preds, label='Predicted', color='red')
            
            # Format x-axis to show dates
            if is_hourly:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
            
            # Automatically rotate date labels for better readability
            fig = plt.gcf()
            fig.autofmt_xdate()
        else:
            # Fallback to index-based plotting if dates are not available
            limit = min(medium_horizon, len(actuals))
            medium_actuals = actuals[:limit, i]
            medium_preds = predictions[:limit, i]
            x_values = range(len(medium_preds))
            
            plt.plot(x_values, medium_actuals, label='Actual')
            plt.plot(x_values, medium_preds, label='Predicted', color='red')
            
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        medium_term_text = "Weekly" if is_hourly else "Monthly"
        title_text = f'{medium_term_text} Forecast for {feature}'
        plt.title(title_text)
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        
        # Save medium-term plot
        filename = f"{feature}_medium_term_{timestamp}.png"
        filepath = os.path.join(VIZ_CONFIG['save_dir'], filename)
        plt.savefig(filepath)
        plt.close()
        
        # 3. Long-term forecast plot (entire test set, to see seasonal patterns)
        if len(actuals) > medium_horizon:
            plt.figure(figsize=(12, 6))
            
            if has_dates and len(dates) == len(actuals):
                plt.plot(dates, actuals[:, i], label='Actual', color='blue')
                plt.plot(dates, predictions[:, i], label='Predicted', color='red')
                
                # Format x-axis to show dates
                if is_hourly:
                    # For hourly data spanning a long period, show month labels
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                
                # Automatically rotate date labels for better readability
                fig = plt.gcf()
                fig.autofmt_xdate()
            else:
                # Fallback to index-based plotting
                x_values = range(len(actuals))
                plt.plot(x_values, actuals[:, i], label='Actual')
                plt.plot(x_values, predictions[:, i], label='Predicted', color='red')
                
                # Add ticks at regular intervals
                plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
            
            title_text = f'Long-term Forecast for {feature} (Full Test Set)'
            plt.title(title_text)
            plt.xlabel('Time')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True)
            
            # Save long-term plot
            filename = f"{feature}_long_term_{timestamp}.png"
            filepath = os.path.join(VIZ_CONFIG['save_dir'], filename)
            plt.savefig(filepath)
            plt.close()
        
        print(f"Saved {feature} plots to {VIZ_CONFIG['save_dir']}/")
    
    # Return timestamp for potential use in loss plot
    return timestamp

def main():
    # Get configuration
    batch_size = MODEL_CONFIG['batch_size']
    num_channels = MODEL_CONFIG['num_channels']
    kernel_size = MODEL_CONFIG['kernel_size']
    dropout = MODEL_CONFIG['dropout']
    dilation_base = MODEL_CONFIG.get('dilation_base', 2)
    
    sequence_length = DATA_CONFIG['sequence_length']
    target_features = DATA_CONFIG['target_features']
    input_features = DATA_CONFIG['input_features']
    is_hourly = DATA_CONFIG.get('is_hourly', False)
    forecast_horizon = DATA_CONFIG.get('forecast_horizon', 24 if is_hourly else 1)
    
    data_type = "hourly" if is_hourly else "daily"
    print(f"\nInitializing for {data_type} data with:")
    print(f"  Sequence length: {sequence_length} time steps")
    print(f"  Forecast horizon: {forecast_horizon} hours ahead" if is_hourly else f"  Predicting {forecast_horizon} days ahead")
    
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
    print(f"  Dilation base: {dilation_base}")
    
    # Initialize model
    model = TemporalConvNet(
        num_inputs=num_inputs, 
        num_outputs=num_outputs, 
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        dilation_base=dilation_base
    )
    
    # Train model
    weight_decay = TRAINING_CONFIG.get('weight_decay', 0)
    train_losses, test_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        weight_decay=weight_decay
    )
    
    # Plot predictions and get timestamp
    timestamp = None
    if VIZ_CONFIG['plot_predictions']:
        timestamp = plot_predictions(model, test_loader, scalers, target_features)
    
    # Plot losses
    if VIZ_CONFIG['plot_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Use the same timestamp for loss plot
        if timestamp:
            loss_filename = f"losses_{timestamp}.png"
        else:
            loss_filename = f"losses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        loss_filepath = os.path.join(VIZ_CONFIG['save_dir'], loss_filename)
        plt.savefig(loss_filepath)
        plt.close()
        print(f"Saved loss plot to {loss_filepath}")

if __name__ == '__main__':
    main() 