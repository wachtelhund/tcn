# Configuration for TCN model and data

# Data configuration
DATA_CONFIG = {
    # Path to training and test data
    'train_file': './DailyDelhiClimateTrain.csv',
    'test_file': './DailyDelhiClimateTest.csv',
    
    # Single file option (alternative to separate train/test files)
    'single_file': None,  # Set to path of single file to use instead of train/test files
    'date_column': 'date',  # Name of date column for splitting
    'train_test_split_date': None,  # Date to split train/test (e.g., '2017-01-01')
    'test_ratio': 0.2,  # Used if no split date provided (ratio of data to use for testing)
    
    # File format options
    'delimiter': ',',  # CSV delimiter (use ';' for semicolon-delimited files)
    'datetime_format': None,  # Format for parsing dates (e.g., '%Y-%m-%d %H:%M' for '2024-01-01 09:00')
    
    # Features configuration
    'target_features': ['meantemp', 'humidity', 'wind_speed', 'meanpressure'],
    
    # If input_features is None, all columns except target_features will be used as inputs
    # If specified, only these features will be used as inputs
    'input_features': ['meantemp', 'humidity', 'wind_speed', 'meanpressure'],
    
    # Time series configuration
    'sequence_length': 7,  # How many time steps to use for prediction (7 days)
}

# Model configuration
MODEL_CONFIG = {
    'batch_size': 32,
    'num_channels': [64, 64, 32, 32],  # Channel sizes for TCN layers
    'kernel_size': 3,                   # Kernel size for TCN
    'dropout': 0.2,                     # Dropout rate
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 200,
    'learning_rate': 0.001,
    'patience': 10,                     # Early stopping patience
    'checkpoint_dir': 'checkpoints',    # Directory to save models
}

# Visualization configuration
VIZ_CONFIG = {
    'plot_predictions': True,
    'plot_losses': True,
    'save_dir': 'plots',
} 