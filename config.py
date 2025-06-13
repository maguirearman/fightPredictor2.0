"""
Configuration settings for MMA Fight Predictor 2.0
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Scraping Configuration
SCRAPING_CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'request_timeout': 10,
    'delay_between_requests': 1.0,  # seconds
    'max_retries': 3,
    'max_events_to_scrape': 50,
    'max_fighters_per_event': 30
}

# Data Sources
DATA_SOURCES = {
    'ufc_stats': 'http://ufcstats.com',
    'sherdog': 'https://www.sherdog.com',
    'ufc_official': 'https://www.ufc.com'
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models': {
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'neural_network': {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': 42
        }
    }
}

# Feature Configuration
FEATURE_CONFIG = {
    'fighter_features': [
        'knockdowns', 'total_strikes_att', 'total_strikes_succ',
        'sig_strikes_att', 'sig_strikes_succ', 'takedown_att',
        'takedown_succ', 'submission_att', 'reversals', 'ctrl_time'
    ],
    'derived_features': [
        'strike_accuracy', 'takedown_accuracy', 'finish_rate',
        'avg_fight_time', 'wins_last_5', 'losses_last_5'
    ]
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'MMA Fight Predictor 2.0',
    'page_icon': '🥊',
    'layout': 'wide',
    'theme': {
        'primary_color': '#FF6B6B',
        'background_color': '#FFFFFF',
        'secondary_background_color': '#F0F2F6',
        'text_color': '#262730'
    }
}

# File Paths
FILE_PATHS = {
    'raw_fights': RAW_DATA_DIR / 'raw_fights.csv',
    'raw_fighters': RAW_DATA_DIR / 'raw_fighters.csv',
    'raw_stats': RAW_DATA_DIR / 'raw_stats.csv',
    'processed_fights': PROCESSED_DATA_DIR / 'processed_fights.csv',
    'processed_fighters': PROCESSED_DATA_DIR / 'processed_fighters.csv',
    'fighter_features': PROCESSED_DATA_DIR / 'fighter_features.csv',
    'training_data': PROCESSED_DATA_DIR / 'training_data.csv',
    'model_gradient_boosting': MODELS_DIR / 'gb_model.pkl',
    'model_random_forest': MODELS_DIR / 'rf_model.pkl',
    'model_neural_network': MODELS_DIR / 'nn_model.pkl',
    'model_metadata': MODELS_DIR / 'model_metadata.json'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'app.log'
}

# Environment Variables (optional)
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 8501))

# Weight Classes (for validation)
WEIGHT_CLASSES = [
    'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
    'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight',
    'Heavyweight', "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight"
]

# Data Validation Rules
VALIDATION_RULES = {
    'fighter_name_min_length': 2,
    'fighter_name_max_length': 50,
    'valid_results': ['Win', 'Loss', 'Draw', 'NC'],
    'valid_methods': [
        'KO/TKO', 'Submission', 'Decision', 'DQ', 'Technical Decision'
    ],
    'max_rounds': 5,
    'min_fight_time': '0:01',
    'max_fight_time': '25:00'
}