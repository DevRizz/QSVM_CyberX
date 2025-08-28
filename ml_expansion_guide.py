"""
Complete Guide for Expanding ML Models in Quantum Anomaly Detection Project
This file shows how to add new models, optimize parameters, and enhance performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import optuna
from river import anomaly, ensemble, drift
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class AdvancedMLExpansion:
    """
    Complete framework for adding and optimizing ML models
    """
    
    def __init__(self):
        self.models = {}
        self.optimized_params = {}
        self.ensemble_models = {}
        
    # ==================== ADDING NEW MODELS ====================
    
    def add_isolation_forest(self):
        """
        Isolation Forest - Excellent for anomaly detection
        Works by isolating anomalies (easier to isolate than normal points)
        """
        model_config = {
            'model': IsolationForest,
            'default_params': {
                'n_estimators': 100,
                'contamination': 0.1,  # Expected proportion of anomalies
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False,
                'random_state': 42
            },
            'param_grid': {
                'n_estimators': [50, 100, 200, 300],
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'max_samples': ['auto', 0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            },
            'optimization_strategy': 'unsupervised'
        }
        self.models['isolation_forest'] = model_config
        return model_config
    
    def add_xgboost_classifier(self):
        """
        XGBoost - State-of-the-art gradient boosting
        Excellent performance, handles missing values, built-in regularization
        """
        model_config = {
            'model': xgb.XGBClassifier,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'eval_metric': 'logloss'
            },
            'param_grid': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 2.0, 5.0]
            },
            'optimization_strategy': 'supervised'
        }
        self.models['xgboost'] = model_config
        return model_config
    
    def add_lightgbm_classifier(self):
        """
        LightGBM - Fast gradient boosting, memory efficient
        Often outperforms XGBoost with faster training
        """
        model_config = {
            'model': lgb.LGBMClassifier,
            'default_params': {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1
            },
            'param_grid': {
                'n_estimators': [100, 200, 300, 500],
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            },
            'optimization_strategy': 'supervised'
        }
        self.models['lightgbm'] = model_config
        return model_config
    
    def add_neural_network(self):
        """
        Multi-layer Perceptron - Deep learning approach
        Can learn complex non-linear patterns
        """
        model_config = {
            'model': MLPClassifier,
            'default_params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42
            },
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25), (200, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'solver': ['adam', 'lbfgs']
            },
            'optimization_strategy': 'supervised'
        }
        self.models['neural_network'] = model_config
        return model_config
    
    def add_local_outlier_factor(self):
        """
        Local Outlier Factor - Density-based anomaly detection
        Good for detecting local anomalies in varying density regions
        """
        model_config = {
            'model': LocalOutlierFactor,
            'default_params': {
                'n_neighbors': 20,
                'algorithm': 'auto',
                'leaf_size': 30,
                'metric': 'minkowski',
                'contamination': 0.1,
                'novelty': True  # For prediction on new data
            },
            'param_grid': {
                'n_neighbors': [10, 15, 20, 25, 30],
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'metric': ['minkowski', 'euclidean', 'manhattan']
            },
            'optimization_strategy': 'unsupervised'
        }
        self.models['local_outlier_factor'] = model_config
        return model_config
    
    def add_autoencoder_anomaly_detector(self):
        """
        Autoencoder - Neural network for anomaly detection
        Learns to reconstruct normal patterns, fails on anomalies
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            
            class AutoencoderAnomalyDetector:
                def __init__(self, input_dim, encoding_dim=32, threshold_percentile=95):
                    self.input_dim = input_dim
                    self.encoding_dim = encoding_dim
                    self.threshold_percentile = threshold_percentile
                    self.autoencoder = None
                    self.threshold = None
                    
                def build_model(self):
                    # Encoder
                    input_layer = Input(shape=(self.input_dim,))
                    encoded = Dense(64, activation='relu')(input_layer)
                    encoded = Dense(self.encoding_dim, activation='relu')(encoded)
                    
                    # Decoder
                    decoded = Dense(64, activation='relu')(encoded)
                    decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
                    
                    # Autoencoder model
                    self.autoencoder = Model(input_layer, decoded)
                    self.autoencoder.compile(optimizer='adam', loss='mse')
                    
                def fit(self, X_normal):
                    if self.autoencoder is None:
                        self.build_model()
                    
                    # Train only on normal data
                    self.autoencoder.fit(X_normal, X_normal, 
                                       epochs=100, batch_size=32, 
                                       validation_split=0.1, verbose=0)
                    
                    # Set threshold based on reconstruction error
                    reconstructions = self.autoencoder.predict(X_normal)
                    mse = np.mean(np.power(X_normal - reconstructions, 2), axis=1)
                    self.threshold = np.percentile(mse, self.threshold_percentile)
                    
                def predict(self, X):
                    reconstructions = self.autoencoder.predict(X)
                    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                    return (mse > self.threshold).astype(int)
                
                def decision_function(self, X):
                    reconstructions = self.autoencoder.predict(X)
                    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                    return mse
            
            model_config = {
                'model': AutoencoderAnomalyDetector,
                'default_params': {
                    'encoding_dim': 32,
                    'threshold_percentile': 95
                },
                'param_grid': {
                    'encoding_dim': [16, 32, 64, 128],
                    'threshold_percentile': [90, 95, 97, 99]
                },
                'optimization_strategy': 'unsupervised'
            }
            self.models['autoencoder'] = model_config
            return model_config
            
        except ImportError:
            print("TensorFlow not available. Skipping autoencoder.")
            return None
    
    # ==================== STREAMING ML MODELS ====================
    
    def add_streaming_models(self):
        """
        Add advanced streaming ML models using River
        """
        streaming_models = {
            'adaptive_random_forest': {
                'model': ensemble.AdaptiveRandomForestClassifier,
                'params': {
                    'n_models': 10,
                    'max_depth': 10,
                    'lambda_value': 6,
                    'performance_metric': 'accuracy'
                }
            },
            'leveraging_bagging': {
                'model': ensemble.LeveragingBaggingClassifier,
                'params': {
                    'n_models': 10,
                    'w': 6,
                    'adwin_delta': 0.002
                }
            },
            'streaming_random_patches': {
                'model': ensemble.SRPClassifier,
                'params': {
                    'n_models': 10,
                    'subspace_size': 0.6,
                    'training_method': 'patches'
                }
            }
        }
        
        for name, config in streaming_models.items():
            self.models[f'streaming_{name}'] = {
                'model': config['model'],
                'default_params': config['params'],
                'optimization_strategy': 'streaming'
            }
        
        return streaming_models

    # ==================== PARAMETER OPTIMIZATION ====================
    
    def optimize_with_grid_search(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """
        Grid Search optimization - Exhaustive search over parameter grid
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.models[model_name]
        model = config['model']()
        param_grid = config['param_grid']
        
        # Custom scoring for imbalanced data
        scorer = make_scorer(f1_score, average='weighted')
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, scoring=scorer, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.optimized_params[model_name] = {
            'method': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def optimize_with_random_search(self, model_name: str, X: np.ndarray, y: np.ndarray, n_iter: int = 100):
        """
        Random Search optimization - More efficient than grid search
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.models[model_name]
        model = config['model']()
        param_grid = config['param_grid']
        
        scorer = make_scorer(f1_score, average='weighted')
        
        random_search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=n_iter, cv=5, scoring=scorer,
            n_jobs=-1, verbose=1, random_state=42
        )
        
        random_search.fit(X, y)
        
        self.optimized_params[model_name] = {
            'method': 'random_search',
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'n_iter': n_iter
        }
        
        return random_search.best_estimator_, random_search.best_params_
    
    def optimize_with_optuna(self, model_name: str, X: np.ndarray, y: np.ndarray, n_trials: int = 100):
        """
        Optuna optimization - Advanced Bayesian optimization
        More efficient than grid/random search
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.models[model_name]
        
        def objective(trial):
            # Define parameter suggestions based on model type
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                    'random_state': 42
                }
            elif model_name == 'neural_network':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                layers = []
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f'layer_{i}_size', 25, 200))
                
                params = {
                    'hidden_layer_sizes': tuple(layers),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                    'max_iter': 500,
                    'random_state': 42
                }
            else:
                # Generic parameter optimization
                params = {}
                for param, values in config['param_grid'].items():
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)
            
            # Train and evaluate model
            model = config['model'](**params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.optimized_params[model_name] = {
            'method': 'optuna',
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'study': study
        }
        
        return study.best_params, study.best_value
    
    # ==================== ENSEMBLE METHODS ====================
    
    def create_voting_ensemble(self, model_names: List[str], X: np.ndarray, y: np.ndarray):
        """
        Create voting ensemble combining multiple models
        """
        estimators = []
        
        for name in model_names:
            if name in self.optimized_params:
                params = self.optimized_params[name]['best_params']
            else:
                params = self.models[name]['default_params']
            
            model = self.models[name]['model'](**params)
            estimators.append((name, model))
        
        # Hard voting for classification
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities for better performance
            n_jobs=-1
        )
        
        voting_clf.fit(X, y)
        self.ensemble_models['voting_ensemble'] = voting_clf
        
        return voting_clf
    
    def create_stacking_ensemble(self, base_models: List[str], meta_model: str, X: np.ndarray, y: np.ndarray):
        """
        Create stacking ensemble with meta-learner
        """
        from sklearn.ensemble import StackingClassifier
        
        # Prepare base estimators
        estimators = []
        for name in base_models:
            if name in self.optimized_params:
                params = self.optimized_params[name]['best_params']
            else:
                params = self.models[name]['default_params']
            
            model = self.models[name]['model'](**params)
            estimators.append((name, model))
        
        # Meta-learner
        if meta_model in self.optimized_params:
            meta_params = self.optimized_params[meta_model]['best_params']
        else:
            meta_params = self.models[meta_model]['default_params']
        
        meta_clf = self.models[meta_model]['model'](**meta_params)
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_clf,
            cv=5,
            n_jobs=-1
        )
        
        stacking_clf.fit(X, y)
        self.ensemble_models['stacking_ensemble'] = stacking_clf
        
        return stacking_clf
    
    # ==================== ADVANCED FEATURES ====================
    
    def add_concept_drift_detection(self):
        """
        Add concept drift detection for streaming data
        """
        drift_detectors = {
            'adwin': drift.ADWIN(delta=0.002),
            'page_hinkley': drift.PageHinkley(min_instances=30, delta=0.005, threshold=50),
            'kswin': drift.KSWIN(alpha=0.005, window_size=100, stat_size=30)
        }
        
        return drift_detectors
    
    def add_feature_selection(self, X: np.ndarray, y: np.ndarray):
        """
        Add automated feature selection
        """
        from sklearn.feature_selection import (
            SelectKBest, f_classif, RFE, SelectFromModel
        )
        from sklearn.ensemble import RandomForestClassifier
        
        feature_selectors = {
            'univariate': SelectKBest(score_func=f_classif, k=10),
            'rfe': RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10),
            'model_based': SelectFromModel(RandomForestClassifier(n_estimators=100))
        }
        
        selected_features = {}
        for name, selector in feature_selectors.items():
            selector.fit(X, y)
            selected_features[name] = {
                'selector': selector,
                'selected_indices': selector.get_support(indices=True),
                'n_features': selector.get_support().sum()
            }
        
        return selected_features
    
    def add_automated_feature_engineering(self, df: pd.DataFrame):
        """
        Add automated feature engineering
        """
        engineered_features = df.copy()
        
        # Polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(df[numeric_cols])
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            
            # Add polynomial features
            for i, name in enumerate(poly_feature_names):
                if name not in numeric_cols:  # Skip original features
                    engineered_features[f'poly_{name}'] = poly_features[:, i]
        
        # Rolling statistics
        for col in numeric_cols:
            engineered_features[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
            engineered_features[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
            engineered_features[f'{col}_lag_1'] = df[col].shift(1)
        
        # Fill NaN values
        engineered_features = engineered_features.fillna(0)
        
        return engineered_features

# ==================== INTEGRATION WITH EXISTING PROJECT ====================

def integrate_new_models_into_project():
    """
    Show how to integrate new models into the existing project structure
    """
    
    integration_code = '''
    # 1. Create new model file: quantum_anomaly/models/advanced_models.py
    
    from .advanced_ml_expansion import AdvancedMLExpansion
    
    class AdvancedModelManager:
        def __init__(self):
            self.ml_expansion = AdvancedMLExpansion()
            self.active_models = {}
            
        def initialize_models(self):
            # Add all new models
            self.ml_expansion.add_xgboost_classifier()
            self.ml_expansion.add_lightgbm_classifier()
            self.ml_expansion.add_isolation_forest()
            self.ml_expansion.add_neural_network()
            
        def train_and_optimize(self, X, y):
            results = {}
            
            for model_name in self.ml_expansion.models:
                print(f"Optimizing {model_name}...")
                
                # Use Optuna for optimization (most efficient)
                best_params, best_score = self.ml_expansion.optimize_with_optuna(
                    model_name, X, y, n_trials=50
                )
                
                # Train final model with best parameters
                model_class = self.ml_expansion.models[model_name]['model']
                final_model = model_class(**best_params)
                final_model.fit(X, y)
                
                self.active_models[model_name] = final_model
                results[model_name] = {'score': best_score, 'params': best_params}
                
            return results
    
    # 2. Modify orchestrator.py to include new models
    
    class EnhancedOrchestrator(Orchestrator):
        def __init__(self):
            super().__init__()
            self.advanced_models = AdvancedModelManager()
            self.advanced_models.initialize_models()
            
        def process_packets_advanced(self, packets):
            # Original processing
            result = self.process_packets(packets)
            
            # Advanced model predictions
            if hasattr(self, 'advanced_models') and self.advanced_models.active_models:
                features = self._select_feature_vector(result['df'])
                
                advanced_predictions = {}
                for name, model in self.advanced_models.active_models.items():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)
                        advanced_predictions[name] = proba[:, 1]  # Anomaly probability
                    else:
                        advanced_predictions[name] = model.decision_function(features)
                
                result['advanced_predictions'] = advanced_predictions
                
                # Ensemble prediction (average of all models)
                if advanced_predictions:
                    ensemble_score = np.mean(list(advanced_predictions.values()), axis=0)
                    result['ensemble_score'] = ensemble_score.tolist()
            
            return result
    
    # 3. Add new Streamlit page for advanced models
    # streamlit_app/pages/09_Advanced_Models.py
    '''
    
    return integration_code

# ==================== PERFORMANCE RECOMMENDATIONS ====================

def get_performance_recommendations():
    """
    Best practices and recommendations for optimal performance
    """
    
    recommendations = {
        'model_selection': {
            'for_small_datasets': ['xgboost', 'lightgbm', 'neural_network'],
            'for_large_datasets': ['isolation_forest', 'streaming_models'],
            'for_imbalanced_data': ['xgboost', 'lightgbm', 'isolation_forest'],
            'for_real_time': ['streaming_models', 'isolation_forest'],
            'for_interpretability': ['xgboost', 'lightgbm', 'random_forest']
        },
        
        'optimization_strategies': {
            'quick_optimization': 'random_search with 50 iterations',
            'thorough_optimization': 'optuna with 200+ trials',
            'production_ready': 'optuna + cross_validation + ensemble'
        },
        
        'ensemble_strategies': {
            'diversity_based': 'Combine different algorithm types (tree-based + neural + kernel)',
            'performance_based': 'Weight models by validation performance',
            'stacking_approach': 'Use meta-learner to combine predictions optimally'
        },
        
        'feature_engineering': {
            'network_specific': 'Add domain knowledge features (port scanning indicators, timing patterns)',
            'automated': 'Use polynomial features and rolling statistics',
            'selection': 'Remove redundant features to improve performance'
        }
    }
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    ml_expansion = AdvancedMLExpansion()
    
    # Add all models
    ml_expansion.add_xgboost_classifier()
    ml_expansion.add_lightgbm_classifier()
    ml_expansion.add_isolation_forest()
    ml_expansion.add_neural_network()
    ml_expansion.add_streaming_models()
    
    print("Available models:", list(ml_expansion.models.keys()))
    print("Integration code generated!")
