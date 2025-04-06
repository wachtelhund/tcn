import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from config import DATA_CONFIG

class CHPDataset(Dataset):
    def __init__(self, csv_file, sequence_length=24, target_features=None, input_features=None):
        """
        Args:
            csv_file (str): Path to the CSV file
            sequence_length (int): Number of time steps to use for prediction
            target_features (list): List of target features to predict
            input_features (list): List of input features to use. If None, all non-target columns are used.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file not found: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.target_features = target_features or DATA_CONFIG['target_features']
        
        # Validate target features exist in the data
        missing_targets = [f for f in self.target_features if f not in self.data.columns]
        if missing_targets:
            raise ValueError(f"Target features not found in data: {missing_targets}")
        
        # Get input features
        if input_features is not None:
            self.input_features = input_features
            # Validate input features exist in the data
            missing_inputs = [f for f in self.input_features if f not in self.data.columns]
            if missing_inputs:
                raise ValueError(f"Input features not found in data: {missing_inputs}")
        else:
            # Use all numeric columns except targets and date
            exclude_columns = self.target_features + ['date']
            self.input_features = [col for col in self.data.columns if col not in exclude_columns]
        
        print(f"Using {len(self.input_features)} input features: {self.input_features}")
        print(f"Predicting {len(self.target_features)} target features: {self.target_features}")
        
        # Normalize the data
        self.scalers = {}
        for feature in self.input_features + self.target_features:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            # Handle constant features (std=0)
            if std == 0:
                std = 1
            self.scalers[feature] = {'mean': mean, 'std': std}
            self.data[feature] = (self.data[feature] - mean) / std

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of input features
        sequence = self.data[self.input_features].iloc[idx:idx + self.sequence_length].values
        
        # Get target values
        target = self.data[self.target_features].iloc[idx + self.sequence_length].values
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

def get_data_loaders(train_file=None, test_file=None, batch_size=32, sequence_length=None, 
                    target_features=None, input_features=None):
    """
    Create data loaders for training and testing.
    
    Args:
        train_file (str): Path to training data CSV
        test_file (str): Path to test data CSV
        batch_size (int): Batch size for training
        sequence_length (int): Number of time steps to use for prediction
        target_features (list): List of target features to predict
        input_features (list): List of input features to use
    
    Returns:
        tuple: (train_loader, test_loader, scalers)
    """
    # Use config values if not specified
    train_file = train_file or DATA_CONFIG['train_file']
    test_file = test_file or DATA_CONFIG['test_file']
    sequence_length = sequence_length or DATA_CONFIG['sequence_length']
    target_features = target_features or DATA_CONFIG['target_features']
    input_features = input_features or DATA_CONFIG['input_features']
    
    print(f"\nLoading data from:")
    print(f"  Train file: {train_file}")
    print(f"  Test file: {test_file}")
    print(f"  Sequence length: {sequence_length}")
    
    train_dataset = CHPDataset(train_file, sequence_length, target_features, input_features)
    test_dataset = CHPDataset(test_file, sequence_length, target_features, input_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.scalers 