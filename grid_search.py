import torch
import numpy as np
import itertools
import os
import json
from datetime import datetime
from tcn import TemporalConvNet
from train import train_model, plot_predictions
from data_loader import get_data_loaders
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

def grid_search(param_grid, metric='test_loss', k_fold=False, n_splits=5):
    """
    Perform grid search over hyperparameter combinations for the TCN model.
    
    Args:
        param_grid (dict): Dictionary with hyperparameter names as keys and lists of parameter values to try.
        metric (str): Metric to optimize ('test_loss', 'rmse', or 'mae').
        k_fold (bool): Whether to use k-fold cross-validation.
        n_splits (int): Number of splits for k-fold cross-validation.
        
    Returns:
        dict: Best parameters and their performance
    """
    # Create results directory
    results_dir = os.path.join(TRAINING_CONFIG['checkpoint_dir'], 'grid_search')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Starting grid search with {len(param_combinations)} parameter combinations")
    
    # Store all results
    all_results = []
    best_score = float('inf')  # Lower is better for our metrics
    best_params = None
    best_model_path = None
    
    # Get data loaders once (will be reused for different model configs)
    print("Preparing data loaders...")
    train_loader, test_loader, scalers = get_data_loaders()
    
    # Get fixed parameters
    num_inputs = len(train_loader.dataset.input_features)
    num_outputs = len(DATA_CONFIG['target_features'])
    
    # Track time
    start_time = datetime.now()
    
    # Run training for each parameter combination
    for i, params in enumerate(param_combinations):
        param_dict = {name: value for name, value in zip(param_names, params)}
        
        print(f"\n[{i+1}/{len(param_combinations)}] Testing parameters: {param_dict}")
        
        # Create model with these parameters
        model_params = {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs
        }
        
        # Add parameters from grid search
        for name, value in param_dict.items():
            if name in ['num_channels', 'kernel_size', 'dropout']:
                model_params[name] = value
        
        # Initialize model
        model = TemporalConvNet(**model_params)
        
        # Extract training parameters
        train_params = {}
        for name, value in param_dict.items():
            if name in ['num_epochs', 'learning_rate', 'patience']:
                train_params[name] = value
        
        # Train model
        train_losses, test_losses = train_model(model, train_loader, test_loader, **train_params)
        
        # Evaluate model
        model.eval()
        if metric == 'test_loss':
            score = min(test_losses)  # Use best test loss
        else:
            # Compute RMSE and MAE metrics on test set
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x.transpose(1, 2))
                    predictions.append(outputs[:, -1].numpy())
                    actuals.append(batch_y.numpy())
            
            predictions = np.concatenate(predictions)
            actuals = np.concatenate(actuals)
            
            # Denormalize predictions and actuals
            target_features = DATA_CONFIG['target_features']
            denorm_predictions = np.zeros_like(predictions)
            denorm_actuals = np.zeros_like(actuals)
            
            for i, feature in enumerate(target_features):
                denorm_predictions[:, i] = predictions[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
                denorm_actuals[:, i] = actuals[:, i] * scalers[feature]['std'] + scalers[feature]['mean']
            
            # Calculate metrics
            if metric == 'rmse':
                errors = []
                for i in range(denorm_predictions.shape[1]):
                    rmse = np.sqrt(np.mean((denorm_predictions[:, i] - denorm_actuals[:, i])**2))
                    errors.append(rmse)
                score = np.mean(errors)  # Average RMSE across all features
            elif metric == 'mae':
                errors = []
                for i in range(denorm_predictions.shape[1]):
                    mae = np.mean(np.abs(denorm_predictions[:, i] - denorm_actuals[:, i]))
                    errors.append(mae)
                score = np.mean(errors)  # Average MAE across all features
        
        # Save results
        result = {
            'params': param_dict,
            'score': score,
            'metric': metric,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        all_results.append(result)
        
        # Update best parameters if this combination is better
        if score < best_score:
            best_score = score
            best_params = param_dict
            best_model_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], 
                                          f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best score: {best_score:.4f} with parameters: {best_params}")
        
        # Save all results to JSON after each combination
        results_file = os.path.join(results_dir, f'grid_search_results_{datetime.now().strftime("%Y%m%d")}.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = []
            for res in all_results:
                ser_res = res.copy()
                ser_res['train_losses'] = [float(x) for x in res['train_losses']]
                ser_res['test_losses'] = [float(x) for x in res['test_losses']]
                serializable_results.append(ser_res)
            
            json.dump({
                'results': serializable_results,
                'best_params': best_params,
                'best_score': float(best_score),
                'best_model_path': best_model_path
            }, f, indent=4)
    
    # Calculate total time
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nGrid search completed in {duration}")
    print(f"Best parameters: {best_params}")
    print(f"Best score ({metric}): {best_score:.4f}")
    print(f"Results saved to {results_file}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_model_path': best_model_path
    }

def run_example_grid_search():
    """
    Example usage of grid search with predefined parameter grid
    """
    # Define parameter grid
    param_grid = {
        # Model parameters
        'num_channels': [
            [32, 32, 16, 16],
            [64, 64, 32, 32],
            [128, 64, 32, 16]
        ],
        'kernel_size': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        
        # Training parameters
        'learning_rate': [0.001, 0.0005, 0.0001],
        'patience': [5, 10, 15]
    }
    
    # Run grid search optimizing for test loss
    results = grid_search(param_grid, metric='test_loss')
    
    # Load and evaluate best model
    best_params = results['best_params']
    best_model_path = results['best_model_path']
    
    # Prepare data loaders
    train_loader, test_loader, scalers = get_data_loaders()
    
    # Get dimensions for model initialization
    num_inputs = len(train_loader.dataset.input_features)
    num_outputs = len(DATA_CONFIG['target_features'])
    
    # Initialize model with best parameters
    model_params = {
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'num_channels': best_params.get('num_channels', MODEL_CONFIG['num_channels']),
        'kernel_size': best_params.get('kernel_size', MODEL_CONFIG['kernel_size']),
        'dropout': best_params.get('dropout', MODEL_CONFIG['dropout'])
    }
    
    # Load best model
    model = TemporalConvNet(**model_params)
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate and plot results
    plot_predictions(model, test_loader, scalers, DATA_CONFIG['target_features'])
    
    return results

if __name__ == "__main__":
    run_example_grid_search() 