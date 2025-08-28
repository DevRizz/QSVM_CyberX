"""
Advanced ML Models Integration for Quantum Anomaly Detection
This file integrates new ML models into the existing project structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from river import anomaly, ensemble
import optuna
from .quantum_svm import QuantumKernelSVC
from .classical_svm import ClassicalSVM
from .online import StreamingBaseline

class AdvancedModelManager:
    """
    Manages advanced ML models for the quantum anomaly detection project
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.optimization_results = {}
        self.ensemble_model = None
        
    def initialize_models(self):
        """Initialize all available models with default parameters"""
        
        # XGBoost - Excellent for structured data
        self.models['xgboost'] = {
            'class': xgb.XGBClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'eval_metric': 'logloss',
                'class_weight': 'balanced'
            }
        }
        
        # LightGBM - Fast and memory efficient
        self.models['lightgbm'] = {
            'class': lgb.LGBMClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
                'class_weight': 'balanced'
            }
        }
        
        # Isolation Forest - Unsupervised anomaly detection
        self.models['isolation_forest'] = {
            'class': IsolationForest,
            'params': {
                'n_estimators': 200,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False,
                'random_state': 42
            }
        }
        
        # Neural Network - Deep learning approach
        self.models['neural_network'] = {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42
            }
        }
        
        # Random Forest - Robust ensemble method
        self.models['random_forest'] = {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42
            }
        }
        
        print(f"Initialized {len(self.models)} advanced models")
    
    def optimize_model_with_optuna(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                                  n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        def objective(trial):
            # Define parameter search space based on model
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'class_weight': 'balanced'
                }
            
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                    'random_state': 42,
                    'verbose': -1,
                    'class_weight': 'balanced'
                }
            
            elif model_name == 'neural_network':
                n_layers = trial.suggest_int('n_layers', 1, 4)
                layers = []
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f'layer_{i}_size', 25, 200))
                
                params = {
                    'hidden_layer_sizes': tuple(layers),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'max_iter': 500,
                    'early_stopping': True,
                    'random_state': 42
                }
            
            elif model_name == 'isolation_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'contamination': trial.suggest_float('contamination', 0.05, 0.2),
                    'max_samples': trial.suggest_categorical('max_samples', ['auto', 0.5, 0.7, 1.0]),
                    'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                    'random_state': 42
                }
            
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            
            # Train and evaluate model
            model = self.models[model_name]['class'](**params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            if model_name == 'isolation_forest':
                # For unsupervised models, use different evaluation
                model.fit(X)
                scores = model.decision_function(X)
                # Convert to binary classification score (simplified)
                return np.mean(scores)
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted', n_jobs=-1)
                return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store results
        self.optimization_results[model_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'study': study
        }
        
        print(f"Optimization completed for {model_name}")
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return self.optimization_results[model_name]
    
    def train_optimized_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all models with optimized parameters
        """
        for model_name in self.models:
            print(f"Training {model_name}...")
            
            # Use optimized parameters if available, otherwise use defaults
            if model_name in self.optimization_results:
                params = self.optimization_results[model_name]['best_params']
            else:
                params = self.models[model_name]['params']
            
            # Train model
            model = self.models[model_name]['class'](**params)
            
            if model_name == 'isolation_forest':
                # Unsupervised training
                model.fit(X)
            else:
                # Supervised training
                model.fit(X, y)
            
            self.trained_models[model_name] = model
            print(f"✓ {model_name} trained successfully")
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray, 
                            model_names: Optional[List[str]] = None):
        """
        Create ensemble model combining multiple trained models
        """
        if model_names is None:
            # Use all supervised models
            model_names = [name for name in self.trained_models 
                          if name != 'isolation_forest']
        
        # Prepare estimators for ensemble
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                estimators.append((name, self.trained_models[name]))
        
        if len(estimators) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        self.ensemble_model.fit(X, y)
        print(f"Ensemble model created with {len(estimators)} models")
        
        return self.ensemble_model
    
    def predict_all_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from all trained models
        """
        predictions = {}
        
        for name, model in self.trained_models.items():
            if name == 'isolation_forest':
                # Anomaly score (higher = more anomalous)
                scores = model.decision_function(X)
                predictions[name] = scores
            else:
                # Classification probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    predictions[name] = proba[:, 1]  # Anomaly probability
                else:
                    pred = model.predict(X)
                    predictions[name] = pred
        
        # Add ensemble prediction if available
        if self.ensemble_model is not None:
            ensemble_proba = self.ensemble_model.predict_proba(X)
            predictions['ensemble'] = ensemble_proba[:, 1]
        
        return predictions
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of model optimization results
        """
        summary = {}
        
        for model_name, results in self.optimization_results.items():
            summary[model_name] = {
                'best_score': results['best_score'],
                'n_trials': results['n_trials'],
                'best_params_count': len(results['best_params'])
            }
        
        return summary

class StreamingAdvancedModels:
    """
    Advanced streaming models using River
    """
    
    def __init__(self):
        self.streaming_models = {}
        self.initialize_streaming_models()
    
    def initialize_streaming_models(self):
        """Initialize streaming models"""
        
        # Adaptive Random Forest
        self.streaming_models['adaptive_rf'] = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_depth=10,
            lambda_value=6,
            performance_metric='accuracy',
            disable_weighted_vote=False,
            drift_detection_method='ADWIN',
            warning_detection_method='ADWIN'
        )
        
        # Leveraging Bagging
        self.streaming_models['leveraging_bagging'] = ensemble.LeveragingBaggingClassifier(
            n_models=10,
            w=6,
            adwin_delta=0.002
        )
        
        # Streaming Random Patches
        self.streaming_models['streaming_patches'] = ensemble.SRPClassifier(
            n_models=10,
            subspace_size=0.6,
            training_method='patches'
        )
    
    def learn_one_all(self, x: Dict[str, float], y: int):
        """Update all streaming models with one sample"""
        for name, model in self.streaming_models.items():
            self.streaming_models[name] = model.learn_one(x, y)
    
    def predict_proba_all(self, x: Dict[str, float]) -> Dict[str, float]:
        """Get predictions from all streaming models"""
        predictions = {}
        
        for name, model in self.streaming_models.items():
            try:
                proba = model.predict_proba_one(x)
                predictions[name] = proba.get(1, 0.0)  # Anomaly probability
            except:
                predictions[name] = 0.5  # Default if prediction fails
        
        return predictions
