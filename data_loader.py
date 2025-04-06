import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from config import DATA_CONFIG

def add_time_features(df, date_column):
    """
    Add time-based features to capture temporal patterns at different scales
    
    Args:
        df (pd.DataFrame): DataFrame with date column
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with added time features
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract time components
    result_df['hour'] = result_df[date_column].dt.hour
    result_df['day'] = result_df[date_column].dt.day
    result_df['day_of_week'] = result_df[date_column].dt.dayofweek  # Monday=0, Sunday=6
    result_df['month'] = result_df[date_column].dt.month
    result_df['year'] = result_df[date_column].dt.year
    
    # Create cyclical features using sine and cosine transformations
    if DATA_CONFIG.get('add_cyclical_features', True):
        # Hour of day (24-hour cycle)
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        
        # Day of week (7-day cycle)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # Month (12-month cycle)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        
        # Day of month normalized (28-31 day cycle)
        days_in_month = result_df[date_column].dt.days_in_month
        result_df['day_of_month_norm'] = result_df['day'] / days_in_month
        result_df['day_of_month_sin'] = np.sin(2 * np.pi * result_df['day_of_month_norm'])
        result_df['day_of_month_cos'] = np.cos(2 * np.pi * result_df['day_of_month_norm'])
    
    return result_df

class CHPDataset(Dataset):
    def __init__(self, data, sequence_length=24, target_features=None, input_features=None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the data
            sequence_length (int): Number of time steps to use for prediction
            target_features (list): List of target features to predict
            input_features (list): List of input features to use. If None, all non-target columns are used.
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.target_features = target_features or DATA_CONFIG['target_features']
        self.is_hourly = DATA_CONFIG.get('is_hourly', False)
        
        # Add time-based features if enabled
        date_column = DATA_CONFIG['date_column']
        if DATA_CONFIG.get('add_time_features', False) and date_column in self.data.columns:
            self.data = add_time_features(self.data, date_column)
        
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
            # Use all numeric columns except targets and date/non-numeric columns
            non_feature_columns = [date_column, 'day', 'day_of_week', 'month', 'year']
            exclude_columns = self.target_features + [col for col in non_feature_columns if col in self.data.columns]
            self.input_features = [col for col in self.data.columns if col not in exclude_columns]
            
            # Add cyclical time features if available
            time_features = [
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos'
            ]
            self.input_features += [f for f in time_features if f in self.data.columns]
        
        data_type = "hourly" if self.is_hourly else "daily"
        print(f"Using {len(self.input_features)} input features: {self.input_features} (data type: {data_type})")
        print(f"Predicting {len(self.target_features)} target features: {self.target_features}")
        
        # Normalize the data
        self.scalers = {}
        for feature in self.input_features + self.target_features:
            # Skip cyclical features that are already normalized
            if any(feature.endswith(suffix) for suffix in ['_sin', '_cos']):
                self.scalers[feature] = {'mean': 0, 'std': 1}
                continue
                
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
        
        # Get target values (next time step after the sequence)
        target = self.data[self.target_features].iloc[idx + self.sequence_length].values
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

def load_and_split_data(file_path, date_column, split_date=None, test_ratio=0.2, 
                       delimiter=',', datetime_format=None, decimal='.'):
    """
    Load data from a single file and split it into train and test sets based on date or ratio
    
    Args:
        file_path (str): Path to the data file
        date_column (str): Name of the date column
        split_date (str, optional): Date to use as split point. If None, test_ratio is used.
        test_ratio (float): Ratio of data to use for testing if split_date is None
        delimiter (str): Delimiter used in the CSV file (e.g., ',' or ';')
        datetime_format (str, optional): Format for parsing dates (e.g., '%Y-%m-%d %H:%M')
        decimal (str): Decimal separator character (e.g., '.' or ',')
        
    Returns:
        tuple: (train_data, test_data)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Load data with appropriate delimiter and decimal separator
    print(f"Loading file with delimiter: '{delimiter}', decimal: '{decimal}'")
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try to diagnose issues with first few lines
        with open(file_path, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        print("First few lines of the file:")
        for i, line in enumerate(first_lines):
            print(f"Line {i+1}: {line.strip()}")
        raise
    
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
                    delimiter=None, datetime_format=None, decimal=None):
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
        decimal (str): Decimal separator character (e.g., '.' or ',')
    
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
    decimal = decimal or DATA_CONFIG['decimal']
    
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
            datetime_format=datetime_format,
            decimal=decimal
        )
        
        # Create datasets
        train_dataset = CHPDataset(train_data, sequence_length, target_features, input_features)
        test_dataset = CHPDataset(test_data, sequence_length, target_features, input_features)
    else:
        print(f"\nLoading data from:")
        print(f"  Train file: {train_file}")
        print(f"  Test file: {test_file}")
        print(f"  Sequence length: {sequence_length}")
        
        # Load train data with appropriate delimiter and decimal
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        train_data = pd.read_csv(train_file, delimiter=delimiter, decimal=decimal)
        
        # Convert date column to datetime if present
        if date_column in train_data.columns:
            if datetime_format:
                train_data[date_column] = pd.to_datetime(train_data[date_column], format=datetime_format)
            else:
                train_data[date_column] = pd.to_datetime(train_data[date_column])
        
        # Load test data with appropriate delimiter and decimal
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        test_data = pd.read_csv(test_file, delimiter=delimiter, decimal=decimal)
        
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