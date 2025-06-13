"""
Machine Learning Model Training Pipeline for MMA Fight Predictor 2.0
Handles multiple algorithms, model evaluation, and persistence
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import MODEL_CONFIG, FILE_PATHS, FEATURE_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MMAModelTrainer:
    """Comprehensive ML model training for MMA fight prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        self.best_model_name = None
        
    def load_training_data(self) -> pd.DataFrame:
        """Load processed training data"""
        try:
            training_df = pd.read_csv(FILE_PATHS['training_data'])
            logger.info(f"Loaded training data: {len(training_df)} samples")
            return training_df
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def prepare_features(self, training_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables for training"""
        # Select feature columns (exclude metadata and target)
        exclude_columns = ['target', 'weight_class', 'fighter1_name', 'fighter2_name', 'event_date']
        feature_columns = [col for col in training_df.columns if col not in exclude_columns]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(training_df[feature_columns])
        y = training_df['target'].values
        
        self.feature_columns = feature_columns
        
        logger.info(f"Prepared features: {X.shape[1]} features, {X.shape[0]} samples")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all ML models with configured parameters"""
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                **MODEL_CONFIG['models']['gradient_boosting']
            ),
            'random_forest': RandomForestClassifier(
                **MODEL_CONFIG['models']['random_forest']
            ),
            'neural_network': MLPClassifier(
                **MODEL_CONFIG['models']['neural_network']
            ),
            'logistic_regression': LogisticRegression(
                random_state=MODEL_CONFIG['random_state'],
                max_iter=1000
            ),
            'svm': SVC(
                probability=True,
                random_state=MODEL_CONFIG['random_state']
            )
        }
        
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, 
                          model_name: str) -> Dict[str, float]:
        """Train and evaluate a single model"""
        logger.info(f"Training {model_name}...")
        
        # Scale features for neural network and SVM
        if model_name in ['neural_network', 'svm', 'logistic_regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=MODEL_CONFIG['cv_folds'], shuffle=True, 
                           random_state=MODEL_CONFIG['random_state'])
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        return metrics
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all models"""
        logger.info("Starting model training pipeline...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        # Initialize models
        models = self.initialize_models()
        
        # Train each model
        performance = {}
        for model_name, model in models.items():
            try:
                metrics = self.train_single_model(
                    model, X_train, y_train, X_test, y_test, model_name
                )
                performance[model_name] = metrics
                self.models[model_name] = model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.model_performance = performance
        
        # Determine best model
        self.best_model_name = max(performance.keys(), 
                                 key=lambda x: performance[x]['accuracy'])
        
        logger.info(f"Best model: {self.best_model_name} "
                   f"(Accuracy: {performance[self.best_model_name]['accuracy']:.4f})")
        
        return performance
    
    def generate_detailed_report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate detailed model performance report"""
        if not self.models:
            raise ValueError("No models have been trained yet")
        
        # Split data for consistent evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        detailed_report = {
            'training_info': {
                'total_samples': len(X),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': X.shape[1],
                'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'timestamp': datetime.now().isoformat()
            },
            'model_performance': self.model_performance,
            'best_model': self.best_model_name,
            'feature_importance': {},
            'classification_reports': {}
        }
        
        # Generate detailed metrics for each model
        for model_name, model in self.models.items():
            try:
                # Prepare data
                if model_name in self.scalers:
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Classification report
                detailed_report['classification_reports'][model_name] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(
                        self.feature_columns,
                        model.feature_importances_
                    ))
                    # Sort by importance
                    sorted_importance = sorted(importance_dict.items(), 
                                             key=lambda x: x[1], reverse=True)
                    detailed_report['feature_importance'][model_name] = sorted_importance[:20]
                
            except Exception as e:
                logger.error(f"Error generating detailed report for {model_name}: {e}")
                continue
        
        return detailed_report
    
    def save_models(self) -> None:
        """Save trained models and metadata"""
        if not self.models:
            raise ValueError("No models to save")
        
        try:
            # Save each model
            for model_name, model in self.models.items():
                model_path = FILE_PATHS['models_dir'] / f'{model_name}_model.pkl'
                joblib.dump(model, model_path)
                
                # Save scaler if exists
                if model_name in self.scalers:
                    scaler_path = FILE_PATHS['models_dir'] / f'{model_name}_scaler.pkl'
                    joblib.dump(self.scalers[model_name], scaler_path)
            
            # Save best model separately for easy loading
            best_model_path = FILE_PATHS['models_dir'] / 'best_model.pkl'
            joblib.dump(self.models[self.best_model_name], best_model_path)
            
            if self.best_model_name in self.scalers:
                best_scaler_path = FILE_PATHS['models_dir'] / 'best_scaler.pkl'
                joblib.dump(self.scalers[self.best_model_name], best_scaler_path)
            
            # Save metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'training_timestamp': datetime.now().isoformat(),
                'models_trained': list(self.models.keys())
            }
            
            with open(FILE_PATHS['model_metadata'], 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(self.models)} models and metadata")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self) -> bool:
        """Load previously trained models"""
        try:
            # Load metadata
            with open(FILE_PATHS['model_metadata'], 'r') as f:
                metadata = json.load(f)
            
            self.best_model_name = metadata['best_model_name']
            self.feature_columns = metadata['feature_columns']
            self.model_performance = metadata['model_performance']
            
            # Load models
            for model_name in metadata['models_trained']:
                model_path = FILE_PATHS['models_dir'] / f'{model_name}_model.pkl'
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    
                    # Load scaler if exists
                    scaler_path = FILE_PATHS['models_dir'] / f'{model_name}_scaler.pkl'
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
            
            logger.info(f"Loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_fight(self, fighter1_stats: Dict, fighter2_stats: Dict, 
                     model_name: str = None) -> Dict[str, float]:
        """Predict fight outcome using trained models"""
        if not self.models:
            raise ValueError("No models loaded. Train or load models first.")
        
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Prepare features
        features = []
        for col in self.feature_columns:
            if col.endswith('_x'):
                base_col = col[:-2]
                features.append(fighter1_stats.get(base_col, 0))
            elif col.endswith('_y'):
                base_col = col[:-2]
                features.append(fighter2_stats.get(base_col, 0))
            else:
                # Handle derived features
                features.append(0)  # Default value
        
        features = np.array(features).reshape(1, -1)
        
        # Scale if necessary
        if model_name in self.scalers:
            features = self.scalers[model_name].transform(features)
        
        # Make prediction
        model = self.models[model_name]
        probabilities = model.predict_proba(features)[0]
        
        return {
            'fighter1_win_prob': probabilities[0],
            'fighter2_win_prob': probabilities[1],
            'predicted_winner': 1 if probabilities[0] > probabilities[1] else 2,
            'confidence': max(probabilities),
            'model_used': model_name
        }
    
    def train_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline"""
        logger.info("Starting complete model training pipeline...")
        
        # Load data
        training_df = self.load_training_data()
        
        # Prepare features
        X, y = self.prepare_features(training_df)
        
        # Train all models
        performance = self.train_all_models(X, y)
        
        # Generate detailed report
        detailed_report = self.generate_detailed_report(X, y)
        
        # Save models
        self.save_models()
        
        logger.info("Model training pipeline completed successfully!")
        
        return detailed_report

class ModelEvaluator:
    """Advanced model evaluation and comparison tools"""
    
    def __init__(self, trainer: MMAModelTrainer):
        self.trainer = trainer
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compare all trained models side by side"""
        if not self.trainer.models:
            raise ValueError("No models to compare")
        
        comparison_data = []
        
        for model_name in self.trainer.models.keys():
            performance = self.trainer.model_performance[model_name]
            comparison_data.append({
                'Model': model_name,
                'Accuracy': performance['accuracy'],
                'Precision': performance['precision'],
                'Recall': performance['recall'],
                'F1-Score': performance['f1'],
                'AUC': performance['auc'],
                'CV Mean': performance['cv_mean'],
                'CV Std': performance['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def feature_importance_analysis(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze feature importance across models"""
        importance_analysis = {}
        
        for model_name, model in self.trainer.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = list(zip(
                    self.trainer.feature_columns,
                    model.feature_importances_
                ))
                importance.sort(key=lambda x: x[1], reverse=True)
                importance_analysis[model_name] = importance[:15]
        
        return importance_analysis

# Convenience functions
def train_mma_models() -> Dict[str, Any]:
    """Train MMA prediction models using the complete pipeline"""
    trainer = MMAModelTrainer()
    return trainer.train_complete_pipeline()

def load_trained_models() -> MMAModelTrainer:
    """Load previously trained models"""
    trainer = MMAModelTrainer()
    success = trainer.load_models()
    if not success:
        raise ValueError("Failed to load models. Train models first.")
    return trainer

if __name__ == "__main__":
    # Test the model trainer
    print("Testing MMA Model Trainer...")
    
    try:
        # Train models
        report = train_mma_models()
        
        print(f"\nTraining Results:")
        print(f"Best model: {report['best_model']}")
        print(f"Models trained: {len(report['model_performance'])}")
        
        for model_name, performance in report['model_performance'].items():
            print(f"{model_name}: {performance['accuracy']:.4f} accuracy")
        
        print("\nModel training test completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have processed training data available.")