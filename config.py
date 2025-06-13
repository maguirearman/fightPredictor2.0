"""
Minimal configuration for MMA Fight Predictor 2.0
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

# Data Sources
DATA_SOURCES = {
    'ufc_stats': 'http://ufcstats.com',
    'sherdog': 'https://www.sherdog.com',
    'ufc_official': 'https://www.ufc.com'
}

# Basic configuration
SCRAPING_CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'request_timeout': 10,
    'delay_between_requests': 1.0,
    'max_retries': 3,
    'max_events_to_scrape': 20,
    'max_fighters_per_event': 30
}

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
        }
    }
}

FEATURE_CONFIG = {
    'fighter_features': [
        'knockdowns', 'total_strikes_att', 'total_strikes_succ',
        'sig_strikes_att', 'sig_strikes_succ', 'takedown_att',
        'takedown_succ', 'submission_att', 'reversals', 'ctrl_time'
    ]
}

STREAMLIT_CONFIG = {
    'page_title': 'MMA Fight Predictor 2.0',
    'page_icon': '🥊',
    'layout': 'wide'
}

# File Paths
FILE_PATHS = {
    'raw_fights': RAW_DATA_DIR / 'raw_fights.csv',
    'raw_fighters': RAW_DATA_DIR / 'raw_fighters.csv',
    'raw_stats': RAW_DATA_DIR / 'raw_stats.csv',
    'processed_fights': PROCESSED_DATA_DIR / 'processed_fights.csv',
    'processed_fighters': PROCESSED_DATA_DIR / 'processed_fighters.csv',
    'training_data': PROCESSED_DATA_DIR / 'training_data.csv',
    'models_dir': MODELS_DIR,
    'model_metadata': MODELS_DIR / 'model_metadata.json'
}

WEIGHT_CLASSES = [
    'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
    'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight',
    'Heavyweight', "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight"
]

VALIDATION_RULES = {
    'fighter_name_min_length': 2,
    'fighter_name_max_length': 50,
    'max_rounds': 5
}