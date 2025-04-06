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
    dates = []
    
    # Check if we have date column in the dataset
    date_column = DATA_CONFIG['date_column']
    has_dates = hasattr(test_loader.dataset, 'data') and date_column in test_loader.dataset.data.columns
    is_hourly = DATA_CONFIG.get('is_hourly', False)
    forecast_horizon = DATA_CONFIG.get('forecast_horizon', 7 if not is_hourly else 24)
    
    with torch.no_grad():
        # Get input data from the last batch for forecasting
        for batch_x, batch_y in test_loader:
            last_batch_x = batch_x
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
        
        # Print metrics
        print(f"\nMetrics for {feature}:")
        print(f"  {forecast_horizon}-step Forecast MSE: {forecast_mse:.4f}")
        print(f"  {forecast_horizon}-step Forecast RMSE: {forecast_rmse:.4f}")
        print(f"  Full Test Set MSE: {full_mse:.4f}")
        print(f"  Full Test Set RMSE: {full_rmse:.4f}")
        
        plt.figure(figsize=(12, 6))
        
        if has_dates and len(dates) == len(actuals):
            # Limit all plots to just the forecast horizon
            limited_dates = dates[:forecast_horizon] if len(dates) > forecast_horizon else dates
            limited_actuals = actuals[:forecast_horizon] if len(actuals) > forecast_horizon else actuals
            limited_preds = predictions[:forecast_horizon] if len(predictions) > forecast_horizon else predictions
            
            # Only plot actual values within the forecast horizon
            plt.plot(limited_dates, limited_actuals[:, i], label='Actual', color='blue')
            
            # Only show predictions for the specified forecast horizon
            unit = "hours" if is_hourly else "days"
            
            if is_hourly:
                # For hourly data
                last_actual_date = dates.iloc[-1] if len(dates) > 0 else None
                
                # Generate future dates
                future_dates = None
                if last_actual_date is not None and (isinstance(last_actual_date, datetime) or pd.api.types.is_datetime64_any_dtype(last_actual_date)):
                    # For hourly data
                    future_dates = [last_actual_date + timedelta(hours=h+1) for h in range(forecast_horizon)]
                
                # Only plot predictions up to the forecast horizon
                plt.plot(limited_dates, limited_preds[:, i], 
                         label=f'Predicted ({forecast_horizon} {unit})', color='red')
                
                # If we have future dates, plot future predictions
                if future_dates and len(future_dates) > 0:
                    # Generate multi-step forecast (same as before)
                    last_x = last_batch_x[-1:].clone()
                    future_preds = []
                    current_input = last_x.clone()
                    
                    for step in range(forecast_horizon):
                        with torch.no_grad():
                            step_output = model(current_input.transpose(1, 2))
                            next_pred = step_output[:, -1]
                        
                        future_preds.append(next_pred.numpy())
                        
                        if step < forecast_horizon - 1:
                            current_input = current_input.clone()
                            current_input = torch.cat([current_input[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
                    
                    future_preds = np.concatenate(future_preds, axis=0)
                    future_preds[:, i] = future_preds[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
                    
                    plt.plot(future_dates, future_preds[:, i], 
                             label=f'Future Forecast ({forecast_horizon} {unit})', color='green', linestyle='-')
            else:
                # For daily data, limit predictions to forecast_horizon days
                # Plot limited predictions on actual dates
                plt.plot(limited_dates, limited_preds[:, i], 
                         label=f'Predicted ({forecast_horizon} {unit})', color='red')
            
            # Format x-axis to show dates
            if is_hourly:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                # Set major ticks to show every few hours
                plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
            else:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                # Set number of ticks based on data length
                if len(limited_dates) > 50:
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(limited_dates)//10)))
                else:
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
            # Automatically rotate date labels for better readability
            fig = plt.gcf()
            fig.autofmt_xdate()
        else:
            # Fallback to index-based plotting if dates are not available
            # Limit to forecast horizon for both actuals and predictions
            limited_actuals = actuals[:forecast_horizon] if len(actuals) > forecast_horizon else actuals
            limited_preds = predictions[:forecast_horizon] if len(predictions) > forecast_horizon else predictions
            x_values = range(len(limited_preds))
            
            plt.plot(x_values, limited_actuals[:, i], label='Actual')
            plt.plot(x_values, limited_preds[:, i], 
                     label=f'Predicted ({forecast_horizon} steps)', color='red')
            
            # Use integer ticks for x-axis
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        title_text = f'Predictions for {feature}'
        title_text += f' ({forecast_horizon}-step Forecast RMSE: {forecast_rmse:.4f})'
        plt.title(title_text)
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        
        # Save with timestamp in filename
        filename = f"{feature}_{timestamp}.png"
        filepath = os.path.join(VIZ_CONFIG['save_dir'], filename)
        plt.savefig(filepath)
        plt.close()
        
        print(f"Saved {feature} plot to {filepath}")
    
    # Return timestamp for potential use in loss plot
    return timestamp

def main():
    # Get configuration
    batch_size = MODEL_CONFIG['batch_size']
    num_channels = MODEL_CONFIG['num_channels']
    kernel_size = MODEL_CONFIG['kernel_size']
    dropout = MODEL_CONFIG['dropout']
    
    sequence_length = DATA_CONFIG['sequence_length']
    target_features = DATA_CONFIG['target_features']
    input_features = DATA_CONFIG['input_features']
    is_hourly = DATA_CONFIG.get('is_hourly', False)
    forecast_horizon = DATA_CONFIG.get('forecast_horizon', 24)
    
    data_type = "hourly" if is_hourly else "daily"
    print(f"\nInitializing for {data_type} data with:")
    print(f"  Sequence length: {sequence_length} time steps")
    print(f"  Forecast horizon: {forecast_horizon} hours ahead" if is_hourly else f"  Predicting next day")
    
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