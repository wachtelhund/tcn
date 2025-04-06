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
    'decimal': '.',  # Decimal separator (use ',' for Swedish/European format)
    
    # Features configuration
    'target_features': ['meantemp', 'humidity', 'wind_speed', 'meanpressure'],
    
    # If input_features is None, all columns except target_features will be used as inputs
    # If specified, only these features will be used as inputs
    'input_features': ['meantemp', 'humidity', 'wind_speed', 'meanpressure'],
    
    # Time series configuration
    'sequence_length': 168,  # How many time steps to use for prediction (7 days * 24 hours)
    'forecast_horizon': 24,  # Number of future time steps to predict (24 hours)
    'is_hourly': True,     # Using hourly data
    
    # Temporal feature extraction
    'add_time_features': True,  # Add hour of day, day of week, month features
    'add_cyclical_features': True,  # Add sin/cos transforms of time features
}

# Model configuration
MODEL_CONFIG = {
    'batch_size': 32,
    'num_channels': [128, 128, 64, 64, 32],  # Deeper channel sizes for TCN layers
    'kernel_size': 7,                   # Larger kernel size to capture daily patterns
    'dropout': 0.2,                     # Reduced dropout for better training stability
    'dilation_base': 2,                 # Base for dilation factor (2^i)
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 200,
    'learning_rate': 0.001,
    'patience': 15,                     # Increased early stopping patience
    'checkpoint_dir': 'checkpoints',    # Directory to save models
    'weight_decay': 1e-6,               # Reduced L2 regularization to prevent underfitting
    'lr_scheduler': 'reduce_plateau',   # Learning rate scheduler type
    'reduce_factor': 0.5,               # Factor by which to reduce learning rate
    'reduce_patience': 7,               # Patience for learning rate scheduler
}

# Visualization configuration
VIZ_CONFIG = {
    'plot_predictions': True,
    'plot_losses': True,
    'save_dir': 'plots',
} 