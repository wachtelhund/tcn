import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from config import DATA_CONFIG

class CHPDataset(Dataset):
    def __init__(self, data, sequence_length=24, target_features=None, input_features=None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the data
            sequence_length (int): Number of time steps to use for prediction
            target_features (list): List of target features to predict
            input_features (list): List of input features to use. If None, all non-target columns are used.
        """
        self.data = data
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
            exclude_columns = self.target_features + [DATA_CONFIG['date_column']]
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

def load_and_split_data(file_path, date_column, split_date=None, test_ratio=0.2, delimiter=',', datetime_format=None):
    """
    Load data from a single file and split it into train and test sets based on date or ratio
    
    Args:
        file_path (str): Path to the data file
        date_column (str): Name of the date column
        split_date (str, optional): Date to use as split point. If None, test_ratio is used.
        test_ratio (float): Ratio of data to use for testing if split_date is None
        delimiter (str): Delimiter used in the CSV file (e.g., ',' or ';')
        datetime_format (str, optional): Format for parsing dates (e.g., '%Y-%m-%d %H:%M')
        
    Returns:
        tuple: (train_data, test_data)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Load data with appropriate delimiter
    print(f"Loading file with delimiter: '{delimiter}'")
    data = pd.read_csv(file_path, delimiter=delimiter)
    
    # Ensure date column exists
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")
    
    # Convert date column to datetime with optional format
    if datetime_format:
        print(f"Parsing dates with format: '{datetime_format}'")
        data[date_column] = pd.to_datetime(data[date_column], format=datetime_format)
    else:
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort by date
    data = data.sort_values(by=date_column)
    
    # Split data based on date or ratio
    if split_date:
        split_date = pd.to_datetime(split_date)
        train_data = data[data[date_column] < split_date].copy()
        test_data = data[data[date_column] >= split_date].copy()
        print(f"Split data by date: {split_date}")
        print(f"  Train data: {len(train_data)} records ({train_data[date_column].min()} to {train_data[date_column].max()})")
        print(f"  Test data: {len(test_data)} records ({test_data[date_column].min()} to {test_data[date_column].max()})")
    else:
        split_idx = int(len(data) * (1 - test_ratio))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        print(f"Split data by ratio: {1-test_ratio:.1f}/{test_ratio:.1f}")
        print(f"  Train data: {len(train_data)} records ({train_data[date_column].min()} to {train_data[date_column].max()})")
        print(f"  Test data: {len(test_data)} records ({test_data[date_column].min()} to {test_data[date_column].max()})")
    
    return train_data, test_data

def get_data_loaders(train_file=None, test_file=None, batch_size=32, sequence_length=None, 
                    target_features=None, input_features=None, single_file=None,
                    date_column=None, train_test_split_date=None, test_ratio=None,
                    delimiter=None, datetime_format=None):
    """
    Create data loaders for training and testing.
    
    Args:
        train_file (str): Path to training data CSV
        test_file (str): Path to test data CSV
        batch_size (int): Batch size for training
        sequence_length (int): Number of time steps to use for prediction
        target_features (list): List of target features to predict
        input_features (list): List of input features to use
        single_file (str): Path to single data file (overrides train_file and test_file)
        date_column (str): Name of date column for splitting
        train_test_split_date (str): Date to split train/test if using single file
        test_ratio (float): Ratio to use for test data if no split date
        delimiter (str): Delimiter used in the CSV file (e.g., ',' or ';')
        datetime_format (str): Format for parsing dates
    
    Returns:
        tuple: (train_loader, test_loader, scalers)
    """
    # Use config values if not specified
    train_file = train_file or DATA_CONFIG['train_file']
    test_file = test_file or DATA_CONFIG['test_file']
    single_file = single_file or DATA_CONFIG['single_file']
    date_column = date_column or DATA_CONFIG['date_column']
    train_test_split_date = train_test_split_date or DATA_CONFIG['train_test_split_date']
    test_ratio = test_ratio or DATA_CONFIG['test_ratio']
    delimiter = delimiter or DATA_CONFIG['delimiter']
    datetime_format = datetime_format or DATA_CONFIG['datetime_format']
    
    sequence_length = sequence_length or DATA_CONFIG['sequence_length']
    target_features = target_features or DATA_CONFIG['target_features']
    input_features = input_features or DATA_CONFIG['input_features']
    
    # Check if we should use a single file with splitting
    if single_file:
        print(f"\nLoading data from single file: {single_file}")
        print(f"  Sequence length: {sequence_length}")
        
        # Load and split data
        train_data, test_data = load_and_split_data(
            single_file, 
            date_column, 
            split_date=train_test_split_date, 
            test_ratio=test_ratio,
            delimiter=delimiter,
            datetime_format=datetime_format
        )
        
        # Create datasets
        train_dataset = CHPDataset(train_data, sequence_length, target_features, input_features)
        test_dataset = CHPDataset(test_data, sequence_length, target_features, input_features)
    else:
        print(f"\nLoading data from:")
        print(f"  Train file: {train_file}")
        print(f"  Test file: {test_file}")
        print(f"  Sequence length: {sequence_length}")
        
        # Load train data with appropriate delimiter
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        train_data = pd.read_csv(train_file, delimiter=delimiter)
        
        # Convert date column to datetime if present
        if date_column in train_data.columns:
            if datetime_format:
                train_data[date_column] = pd.to_datetime(train_data[date_column], format=datetime_format)
            else:
                train_data[date_column] = pd.to_datetime(train_data[date_column])
        
        # Load test data with appropriate delimiter
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        test_data = pd.read_csv(test_file, delimiter=delimiter)
        
        # Convert date column to datetime if present
        if date_column in test_data.columns:
            if datetime_format:
                test_data[date_column] = pd.to_datetime(test_data[date_column], format=datetime_format)
            else:
                test_data[date_column] = pd.to_datetime(test_data[date_column])
        
        # Create datasets
        train_dataset = CHPDataset(train_data, sequence_length, target_features, input_features)
        test_dataset = CHPDataset(test_data, sequence_length, target_features, input_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.scalers 