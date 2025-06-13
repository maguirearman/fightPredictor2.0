"""
Data Processing Pipeline for MMA Fight Predictor 2.0
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from config import FILE_PATHS, FEATURE_CONFIG, VALIDATION_RULES, WEIGHT_CLASSES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MMADataProcessor:
    """Comprehensive data processing for MMA data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw scraped data"""
        try:
            fights_df = pd.read_csv(FILE_PATHS['raw_fights'])
            fighters_df = pd.read_csv(FILE_PATHS['raw_fighters'])
            stats_df = pd.read_csv(FILE_PATHS['raw_stats'])
            
            logger.info(f"Loaded raw data: {len(fights_df)} fights, {len(fighters_df)} fighters, {len(stats_df)} stats")
            return {
                'fights': fights_df,
                'fighters': fighters_df,
                'stats': stats_df
            }
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    
    def clean_fighter_data(self, fighters_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fighter data"""
        df = fighters_df.copy()
        
        # Clean fighter names
        df['name'] = df['name'].str.strip()
        df['name'] = df['name'].str.title()
        
        # Standardize nicknames
        df['nickname'] = df['nickname'].fillna('').str.strip()
        
        # Clean height data (convert to inches)
        df['height_inches'] = df['height'].apply(self._parse_height)
        
        # Clean weight data (extract numeric weight)
        df['weight_lbs'] = df['weight'].apply(self._parse_weight)
        
        # Clean reach data
        df['reach_inches'] = df['reach'].apply(self._parse_reach)
        
        # Parse age
        df['age'] = df['age'].apply(self._parse_age)
        
        # Standardize weight classes
        df['weight_class'] = df['weight_class'].apply(self._standardize_weight_class)
        
        # Calculate win percentage
        df['win_percentage'] = df['wins_total'] / (df['wins_total'] + df['losses_total'])
        df['win_percentage'] = df['win_percentage'].fillna(0)
        
        # Calculate finish rate
        df['finish_rate'] = (df['wins_ko'] + df['wins_sub']) / df['wins_total']
        df['finish_rate'] = df['finish_rate'].fillna(0)
        
        # Calculate submission rate
        df['submission_rate'] = df['wins_sub'] / df['wins_total']
        df['submission_rate'] = df['submission_rate'].fillna(0)
        
        # Remove duplicate fighters (keep most recent record)
        df = df.drop_duplicates(subset=['name'], keep='last')
        
        # Validate data
        df = self._validate_fighter_data(df)
        
        logger.info(f"Cleaned fighter data: {len(df)} fighters remaining")
        return df
    
    def clean_fight_data(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fight data"""
        df = fights_df.copy()
        
        # Clean fighter names
        df['fighter1'] = df['fighter1'].str.strip().str.title()
        df['fighter2'] = df['fighter2'].str.strip().str.title()
        df['winner'] = df['winner'].str.strip().str.title()
        
        # Parse event dates
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        
        # Standardize weight classes
        df['weight_class'] = df['weight_class'].apply(self._standardize_weight_class)
        
        # Clean method data
        df['method'] = df['method'].apply(self._standardize_method)
        
        # Parse fight time
        df['fight_time_seconds'] = df['time'].apply(self._parse_fight_time)
        
        # Parse round
        df['round_finished'] = df['round_finished'].apply(self._parse_round)
        
        # Create binary winner column (1 if fighter1 wins, 0 if fighter2 wins)
        df['fighter1_wins'] = (df['winner'] == df['fighter1']).astype(int)
        
        # Remove fights with missing critical data
        df = df.dropna(subset=['fighter1', 'fighter2', 'winner'])
        
        # Remove fights where fighter1 == fighter2
        df = df[df['fighter1'] != df['fighter2']]
        
        # Validate data
        df = self._validate_fight_data(df)
        
        logger.info(f"Cleaned fight data: {len(df)} fights remaining")
        return df
    
    def clean_stats_data(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fight statistics"""
        df = stats_df.copy()
        
        # Clean fighter names
        df['fighter_name'] = df['fighter_name'].str.strip().str.title()
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            'knockdowns', 'sig_strikes_landed', 'sig_strikes_attempted',
            'total_strikes_landed', 'total_strikes_attempted',
            'takedowns_landed', 'takedowns_attempted', 'submission_attempts',
            'control_time_seconds'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate derived statistics
        df['striking_accuracy'] = np.where(
            df['sig_strikes_attempted'] > 0,
            df['sig_strikes_landed'] / df['sig_strikes_attempted'],
            0
        )
        
        df['takedown_accuracy'] = np.where(
            df['takedowns_attempted'] > 0,
            df['takedowns_landed'] / df['takedowns_attempted'],
            0
        )
        
        df['total_striking_accuracy'] = np.where(
            df['total_strikes_attempted'] > 0,
            df['total_strikes_landed'] / df['total_strikes_attempted'],
            0
        )
        
        # Control time per minute
        df['control_time_per_minute'] = df['control_time_seconds'] / 60
        
        logger.info(f"Cleaned stats data: {len(df)} stat records")
        return df
    
    def aggregate_fighter_stats(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fighter statistics across all fights"""
        # Group by fighter and calculate averages
        aggregated = stats_df.groupby('fighter_name').agg({
            'knockdowns': 'mean',
            'sig_strikes_landed': 'mean',
            'sig_strikes_attempted': 'mean',
            'total_strikes_landed': 'mean',
            'total_strikes_attempted': 'mean',
            'takedowns_landed': 'mean',
            'takedowns_attempted': 'mean',
            'submission_attempts': 'mean',
            'control_time_seconds': 'mean',
            'striking_accuracy': 'mean',
            'takedown_accuracy': 'mean',
            'total_striking_accuracy': 'mean',
            'control_time_per_minute': 'mean'
        }).reset_index()
        
        # Add fight count
        fight_counts = stats_df.groupby('fighter_name').size().reset_index(name='total_fights')
        aggregated = aggregated.merge(fight_counts, on='fighter_name')
        
        # Add recent performance (last 5 fights)
        recent_stats = self._calculate_recent_performance(stats_df)
        aggregated = aggregated.merge(recent_stats, on='fighter_name', how='left')
        
        logger.info(f"Aggregated stats for {len(aggregated)} fighters")
        return aggregated
    
    def create_training_data(self, fights_df: pd.DataFrame, fighter_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Create training dataset by combining fight and fighter data"""
        training_data = []
        
        for _, fight in fights_df.iterrows():
            # Get stats for both fighters
            f1_stats = fighter_stats_df[fighter_stats_df['fighter_name'] == fight['fighter1']]
            f2_stats = fighter_stats_df[fighter_stats_df['fighter_name'] == fight['fighter2']]
            
            if len(f1_stats) == 0 or len(f2_stats) == 0:
                continue
            
            f1_stats = f1_stats.iloc[0]
            f2_stats = f2_stats.iloc[0]
            
            # Create feature row
            features = {}
            
            # Fighter 1 features (suffix _x)
            for col in FEATURE_CONFIG['fighter_features']:
                if col in f1_stats:
                    features[f'{col}_x'] = f1_stats[col]
                else:
                    features[f'{col}_x'] = 0
            
            # Fighter 2 features (suffix _y)
            for col in FEATURE_CONFIG['fighter_features']:
                if col in f2_stats:
                    features[f'{col}_y'] = f2_stats[col]
                else:
                    features[f'{col}_y'] = 0
            
            # Add fight metadata
            features['weight_class'] = fight['weight_class']
            features['fighter1_name'] = fight['fighter1']
            features['fighter2_name'] = fight['fighter2']
            features['event_date'] = fight['event_date']
            
            # Target variable (1 if fighter1 wins, 0 if fighter2 wins)
            features['target'] = fight['fighter1_wins']
            
            # Add derived features
            features.update(self._create_derived_features(f1_stats, f2_stats))
            
            training_data.append(features)
        
        training_df = pd.DataFrame(training_data)
        
        # Handle missing values
        numeric_columns = training_df.select_dtypes(include=[np.number]).columns
        training_df[numeric_columns] = self.imputer.fit_transform(training_df[numeric_columns])
        
        logger.info(f"Created training dataset with {len(training_df)} samples")
        return training_df
    
    def _create_derived_features(self, f1_stats: pd.Series, f2_stats: pd.Series) -> Dict:
        """Create derived features comparing two fighters"""
        features = {}
        
        # Statistical differences
        if f1_stats['sig_strikes_attempted'] > 0 and f2_stats['sig_strikes_attempted'] > 0:
            features['striking_accuracy_diff'] = f1_stats['striking_accuracy'] - f2_stats['striking_accuracy']
        else:
            features['striking_accuracy_diff'] = 0
        
        if f1_stats['takedowns_attempted'] > 0 and f2_stats['takedowns_attempted'] > 0:
            features['takedown_accuracy_diff'] = f1_stats['takedown_accuracy'] - f2_stats['takedown_accuracy']
        else:
            features['takedown_accuracy_diff'] = 0
        
        # Experience difference
        features['experience_diff'] = f1_stats.get('total_fights', 0) - f2_stats.get('total_fights', 0)
        
        # Strike volume difference
        features['strike_volume_diff'] = f1_stats['sig_strikes_attempted'] - f2_stats['sig_strikes_attempted']
        
        # Control time difference
        features['control_time_diff'] = f1_stats['control_time_per_minute'] - f2_stats['control_time_per_minute']
        
        return features
    
    def _calculate_recent_performance(self, stats_df: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
        """Calculate recent performance metrics"""
        # This is simplified - in real implementation, you'd need fight dates
        recent_stats = stats_df.groupby('fighter_name').tail(last_n).groupby('fighter_name').agg({
            'striking_accuracy': 'mean',
            'takedown_accuracy': 'mean',
            'knockdowns': 'sum',
            'submission_attempts': 'sum'
        }).reset_index()
        
        # Rename columns to indicate recent performance
        recent_stats.columns = ['fighter_name'] + [f'recent_{col}' for col in recent_stats.columns[1:]]
        
        return recent_stats
    
    def _parse_height(self, height_str: str) -> float:
        """Parse height string to inches"""
        if pd.isna(height_str) or height_str == "":
            return np.nan
        
        try:
            # Handle formats like "6'2\"" or "6 ft 2 in"
            height_str = str(height_str).replace('"', '').replace('ft', '').replace('in', '')
            
            if "'" in height_str:
                parts = height_str.split("'")
                feet = int(parts[0])
                inches = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 0
                return feet * 12 + inches
            else:
                # Try to extract just the inches
                numbers = re.findall(r'\d+', height_str)
                if len(numbers) >= 2:
                    return int(numbers[0]) * 12 + int(numbers[1])
                elif len(numbers) == 1:
                    return int(numbers[0])
        except:
            pass
        
        return np.nan
    
    def _parse_weight(self, weight_str: str) -> float:
        """Parse weight string to pounds"""
        if pd.isna(weight_str) or weight_str == "":
            return np.nan
        
        try:
            # Extract numeric weight
            numbers = re.findall(r'\d+\.?\d*', str(weight_str))
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        return np.nan
    
    def _parse_reach(self, reach_str: str) -> float:
        """Parse reach string to inches"""
        if pd.isna(reach_str) or reach_str == "":
            return np.nan
        
        try:
            numbers = re.findall(r'\d+\.?\d*', str(reach_str))
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        return np.nan
    
    def _parse_age(self, age_str: str) -> float:
        """Parse age string to numeric age"""
        if pd.isna(age_str) or age_str == "":
            return np.nan
        
        try:
            numbers = re.findall(r'\d+', str(age_str))
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        return np.nan
    
    def _parse_fight_time(self, time_str: str) -> float:
        """Parse fight time to total seconds"""
        if pd.isna(time_str) or time_str == "":
            return np.nan
        
        try:
            # Handle formats like "4:15" or "2:30"
            if ":" in str(time_str):
                parts = str(time_str).split(":")
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except:
            pass
        
        return np.nan
    
    def _parse_round(self, round_str: str) -> int:
        """Parse round string to integer"""
        if pd.isna(round_str) or round_str == "":
            return 3  # Default to 3 rounds
        
        try:
            numbers = re.findall(r'\d+', str(round_str))
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return 3
    
    def _standardize_weight_class(self, weight_class: str) -> str:
        """Standardize weight class names"""
        if pd.isna(weight_class) or weight_class == "":
            return "Unknown"
        
        weight_class = str(weight_class).strip()
        
        # Mapping for common variations
        weight_class_mapping = {
            'Heavyweight': 'Heavyweight',
            'Light Heavyweight': 'Light Heavyweight',
            'Middleweight': 'Middleweight',
            'Welterweight': 'Welterweight',
            'Lightweight': 'Lightweight',
            'Featherweight': 'Featherweight',
            'Bantamweight': 'Bantamweight',
            'Flyweight': 'Flyweight',
            'Women\'s Bantamweight': 'Women\'s Bantamweight',
            'Women\'s Flyweight': 'Women\'s Flyweight',
            'Women\'s Strawweight': 'Women\'s Strawweight',
            'Women\'s Featherweight': 'Women\'s Featherweight'
        }
        
        # Try exact match first
        for standard, mapped in weight_class_mapping.items():
            if standard.lower() in weight_class.lower():
                return mapped
        
        return weight_class
    
    def _standardize_method(self, method: str) -> str:
        """Standardize fight method names"""
        if pd.isna(method) or method == "":
            return "Decision"
        
        method = str(method).lower().strip()
        
        if any(word in method for word in ['ko', 'knockout', 'tko', 'technical knockout']):
            return 'KO/TKO'
        elif any(word in method for word in ['submission', 'sub', 'tap']):
            return 'Submission'
        elif any(word in method for word in ['decision', 'dec']):
            return 'Decision'
        elif any(word in method for word in ['dq', 'disqualification']):
            return 'DQ'
        else:
            return 'Decision'
    
    def _validate_fighter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate fighter data and remove invalid records"""
        initial_count = len(df)
        
        # Remove fighters with invalid names
        df = df[df['name'].str.len() >= VALIDATION_RULES['fighter_name_min_length']]
        df = df[df['name'].str.len() <= VALIDATION_RULES['fighter_name_max_length']]
        
        # Remove fighters with impossible physical stats
        df = df[(df['height_inches'].isna()) | (df['height_inches'].between(60, 84))]  # 5'0" to 7'0"
        df = df[(df['weight_lbs'].isna()) | (df['weight_lbs'].between(100, 300))]  # 100-300 lbs
        df = df[(df['age'].isna()) | (df['age'].between(18, 50))]  # 18-50 years old
        
        logger.info(f"Fighter validation: {initial_count} -> {len(df)} ({initial_count - len(df)} removed)")
        return df
    
    def _validate_fight_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate fight data and remove invalid records"""
        initial_count = len(df)
        
        # Remove fights with invalid rounds
        df = df[df['round_finished'].between(1, VALIDATION_RULES['max_rounds'])]
        
        # Remove fights with invalid weight classes
        valid_weight_classes = WEIGHT_CLASSES + ['Unknown']
        df = df[df['weight_class'].isin(valid_weight_classes)]
        
        logger.info(f"Fight validation: {initial_count} -> {len(df)} ({initial_count - len(df)} removed)")
        return df
    
    def save_processed_data(self, fighters_df: pd.DataFrame, fights_df: pd.DataFrame, 
                           training_df: pd.DataFrame) -> None:
        """Save processed data to files"""
        try:
            fighters_df.to_csv(FILE_PATHS['processed_fighters'], index=False)
            fights_df.to_csv(FILE_PATHS['processed_fights'], index=False)
            training_df.to_csv(FILE_PATHS['training_data'], index=False)
            
            logger.info("Processed data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """Main method to process all data"""
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Clean individual datasets
        clean_fighters = self.clean_fighter_data(raw_data['fighters'])
        clean_fights = self.clean_fight_data(raw_data['fights'])
        clean_stats = self.clean_stats_data(raw_data['stats'])
        
        # Aggregate fighter statistics
        fighter_stats = self.aggregate_fighter_stats(clean_stats)
        
        # Create training data
        training_data = self.create_training_data(clean_fights, fighter_stats)
        
        # Save processed data
        self.save_processed_data(clean_fighters, clean_fights, training_data)
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'fighters': clean_fighters,
            'fights': clean_fights,
            'stats': clean_stats,
            'fighter_stats': fighter_stats,
            'training_data': training_data
        }

# Convenience function
def process_mma_data() -> Dict[str, pd.DataFrame]:
    """Process MMA data using the complete pipeline"""
    processor = MMADataProcessor()
    return processor.process_all_data()

if __name__ == "__main__":
    # Test the data processor
    print("Testing MMA Data Processor...")
    
    try:
        processed_data = process_mma_data()
        
        print(f"\nProcessing Results:")
        for key, df in processed_data.items():
            print(f"{key}: {len(df)} records")
        
        print("\nData processing test completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Make sure you have raw data files available.")