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
    
    # Store indices for accessing raw data later
    indices = []
    
    with torch.no_grad():
        # Get input data from the last batch for forecasting
        idx_offset = test_loader.dataset.sequence_length
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            # Keep track of indices for this batch
            if hasattr(test_loader.dataset, 'raw_data'):
                batch_size = batch_x.shape[0]
                start_idx = batch_idx * test_loader.batch_size + idx_offset
                end_idx = start_idx + batch_size
                indices.extend(list(range(start_idx, end_idx)))
            
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
    
    # Denormalize the predictions and actuals for metrics calculation
    denorm_predictions = np.zeros_like(predictions)
    denorm_actuals = np.zeros_like(actuals)
    
    for i, feature in enumerate(target_features):
        denorm_predictions[:, i] = predictions[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
        denorm_actuals[:, i] = actuals[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
    
    # Get raw data for visualization if available
    raw_actuals = None
    if hasattr(test_loader.dataset, 'raw_data') and indices:
        raw_actuals = np.zeros_like(denorm_actuals)
        for i, feature in enumerate(target_features):
            raw_values = test_loader.dataset.raw_data[feature].values[indices]
            if len(raw_values) == len(denorm_actuals):
                raw_actuals[:, i] = raw_values
                # Apply the same exact transformation to predictions to match raw values scale
                # This ensures predictions are on the same scale as raw actuals
                if raw_values.mean() != denorm_actuals[:, i].mean() or raw_values.std() != denorm_actuals[:, i].std():
                    # Calculate the adjustment factor from denormalized to raw values
                    denorm_mean = denorm_actuals[:, i].mean()
                    denorm_std = denorm_actuals[:, i].std() if denorm_actuals[:, i].std() > 0 else 1
                    raw_mean = raw_values.mean()
                    raw_std = raw_values.std() if raw_values.std() > 0 else 1
                    
                    # Adjust predictions to match the raw data scale
                    denorm_predictions[:, i] = ((denorm_predictions[:, i] - denorm_mean) / denorm_std * raw_std) + raw_mean
            else:
                # Fallback to denormalized values if sizes don't match
                raw_actuals = denorm_actuals
                break
    else:
        # Use denormalized values if raw data is not available
        raw_actuals = denorm_actuals
    
    # Create plots directory
    os.makedirs(VIZ_CONFIG['save_dir'], exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a separate plot for each target feature
    for i, feature in enumerate(target_features):
        # Calculate MSE and RMSE for the limited forecast horizon, not all predictions
        # Use only as many predictions as specified by forecast_horizon
        if len(denorm_actuals) > forecast_horizon:
            forecast_mse = np.mean((denorm_predictions[:forecast_horizon, i] - denorm_actuals[:forecast_horizon, i])**2)
            forecast_rmse = np.sqrt(forecast_mse)
        else:
            forecast_mse = np.mean((denorm_predictions[:, i] - denorm_actuals[:, i])**2)
            forecast_rmse = np.sqrt(forecast_mse)
        
        # Calculate metrics for all predictions (for comparison)
        full_mse = np.mean((denorm_predictions[:, i] - denorm_actuals[:, i])**2)
        full_rmse = np.sqrt(full_mse)
        
        # Print metrics
        print(f"\nMetrics for {feature}:")
        print(f"  {forecast_horizon}-step Forecast MSE: {forecast_mse:.4f}")
        print(f"  {forecast_horizon}-step Forecast RMSE: {forecast_rmse:.4f}")
        print(f"  Full Test Set MSE: {full_mse:.4f}")
        print(f"  Full Test Set RMSE: {full_rmse:.4f}")
        
        plt.figure(figsize=(12, 6))
        
        if has_dates and len(dates) == len(raw_actuals):
            # Only show the forecast horizon in the plot
            limited_dates = dates[:forecast_horizon] if len(dates) > forecast_horizon else dates
            limited_actuals = raw_actuals[:forecast_horizon] if len(raw_actuals) > forecast_horizon else raw_actuals
            limited_preds = denorm_predictions[:forecast_horizon] if len(denorm_predictions) > forecast_horizon else denorm_predictions
            
            unit = "hours" if is_hourly else "days"
            
            if is_hourly:
                # For hourly data, we'll only show the immediate prediction, not the future forecast
                # This simplifies the plot and avoids the issue with the future forecast being too far away
                plt.plot(limited_dates, limited_actuals[:, i], label='Actual', color='blue')
                plt.plot(limited_dates, limited_preds[:, i], label=f'Predicted ({forecast_horizon} {unit})', color='red')
            else:
                # For daily data
                plt.plot(limited_dates, limited_actuals[:, i], label='Actual', color='blue')
                plt.plot(limited_dates, limited_preds[:, i], label=f'Predicted ({forecast_horizon} {unit})', color='red')
            
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
            limited_actuals = raw_actuals[:forecast_horizon] if len(raw_actuals) > forecast_horizon else raw_actuals
            limited_preds = denorm_predictions[:forecast_horizon] if len(denorm_predictions) > forecast_horizon else denorm_predictions
            x_values = range(len(limited_preds))
            
            plt.plot(x_values, limited_actuals[:, i], label='Actual')
            plt.plot(x_values, limited_preds[:, i], label=f'Predicted ({forecast_horizon} steps)', color='red')
            
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
    
    # Create a combined plot with all sensors
    plt.figure(figsize=(14, 8))
    
    # Define colors for different sensors
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Measure inference time
    print("Measuring inference speed...")
    
    # Measure full model inference (all sensors)
    test_batch = next(iter(test_loader))[0]  # Get a single batch
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_batch.transpose(1, 2))
    
    # Full model timing (all sensors)
    start_time = datetime.now()
    with torch.no_grad():
        for _ in range(100):
            outputs = model(test_batch.transpose(1, 2))
    end_time = datetime.now()
    full_inference_time = (end_time - start_time).total_seconds() / 100  # Average time per inference
    print(f"Average inference time (all sensors): {full_inference_time*1000:.2f} ms")
    
    # Per-sensor inference timing
    for i, feature in enumerate(target_features):
        # Create a simple wrapper to time just the extraction of one sensor's output
        def time_single_sensor():
            with torch.no_grad():
                output = model(test_batch.transpose(1, 2))
                return output[:, -1, i]  # Return just one sensor's output
        
        # Timing
        start_time = datetime.now()
        for _ in range(100):
            _ = time_single_sensor()
        end_time = datetime.now()
        sensor_time = (end_time - start_time).total_seconds() / 100
        print(f"Average inference time (Sensor {i+1}): {sensor_time*1000:.2f} ms")
    
    if has_dates and len(dates) == len(raw_actuals):
        # Only show the forecast horizon in the plot
        limited_dates = dates[:forecast_horizon] if len(dates) > forecast_horizon else dates
        limited_actuals = raw_actuals[:forecast_horizon] if len(raw_actuals) > forecast_horizon else raw_actuals
        limited_preds = denorm_predictions[:forecast_horizon] if len(denorm_predictions) > forecast_horizon else denorm_predictions
        
        # Plot each sensor with its actual and forecast
        for i, feature in enumerate(target_features):
            color_idx = i % len(colors)
            # Actual as solid line
            plt.plot(limited_dates, limited_actuals[:, i], 
                     label=f'Actual Sensor {i+1}', 
                     color=colors[color_idx], 
                     linewidth=2)
            
            # Forecast as dotted line
            plt.plot(limited_dates, limited_preds[:, i], 
                     label=f'Forecast Sensor {i+1}', 
                     color=colors[color_idx], 
                     linestyle='--', 
                     linewidth=2)
        
        # Format x-axis to show dates
        if is_hourly:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            if len(limited_dates) > 50:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(limited_dates)//10)))
            else:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        # Automatically rotate date labels
        fig = plt.gcf()
        fig.autofmt_xdate()
    else:
        # Fallback to index-based plotting
        limited_actuals = raw_actuals[:forecast_horizon] if len(raw_actuals) > forecast_horizon else raw_actuals
        limited_preds = denorm_predictions[:forecast_horizon] if len(denorm_predictions) > forecast_horizon else denorm_predictions
        x_values = range(len(limited_preds))
        
        # Plot each sensor with its actual and forecast
        for i, feature in enumerate(target_features):
            color_idx = i % len(colors)
            # Actual as solid line
            plt.plot(x_values, limited_actuals[:, i], 
                     label=f'Actual Sensor {i+1}', 
                     color=colors[color_idx], 
                     linewidth=2)
            
            # Forecast as dotted line
            plt.plot(x_values, limited_preds[:, i], 
                     label=f'Forecast Sensor {i+1}', 
                     color=colors[color_idx], 
                     linestyle='--', 
                     linewidth=2)
        
        # Use integer ticks for x-axis
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title(f'Actual vs. Forecasted Values (TCN) - 1 Day Ahead')
    plt.xlabel('Time')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.grid(True)
    
    # Save the combined plot
    combined_filename = f"combined_sensors_{timestamp}.png"
    combined_filepath = os.path.join(VIZ_CONFIG['save_dir'], combined_filename)
    plt.savefig(combined_filepath)
    plt.close()
    
    print(f"Saved combined sensors plot to {combined_filepath}")
    
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