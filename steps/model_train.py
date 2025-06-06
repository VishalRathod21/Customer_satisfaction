import logging
from typing import Tuple, Dict, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

from .config import ModelNameConfig, ModelType, MODEL_PARAMS

# Configure logging
logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


def create_model(
    model_type: ModelType, 
    params: Dict[str, Any] = None
) -> RegressorMixin:
    """Create a model instance with the given parameters.
    
    Args:
        model_type: Type of the model to create
        params: Model parameters
        
    Returns:
        Initialized model instance
    """
    if params is None:
        params = {}
    
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestRegressor(random_state=42, **params)
    elif model_type == ModelType.XGBOOST:
        return xgb.XGBRegressor(random_state=42, **params)
    elif model_type == ModelType.LIGHTGBM:
        return lgb.LGBMRegressor(random_state=42, **params)
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingRegressor(random_state=42, **params)
    elif model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def tune_hyperparameters(
    model_type: ModelType,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform hyperparameter tuning using Optuna.
    
    Args:
        model_type: Type of the model to tune
        x_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with best hyperparameters
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            params = {}
            param_space = MODEL_PARAMS[model_type]
            
            # Sample parameters
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(x, bool) for x in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(x, int) for x in param_values):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif all(isinstance(x, float) for x in param_values):
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values), log=True
                        )
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values
            
            # Create and evaluate model
            model = create_model(model_type, params)
            score = cross_val_score(
                model, x_train, y_train, 
                cv=cv, scoring='r2', n_jobs=-1
            ).mean()
            
            return score
        
        # Optimize
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
        
    except ImportError:
        logger.warning(
            "Optuna not installed. Using default hyperparameters. "
            "Install with: pip install optuna"
        )
        return {}


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """Train a machine learning model with optional hyperparameter tuning.
    
    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training target
        y_test: Test target
        config: Model configuration
        
    Returns:
        Trained model
    """
    try:
        # Log model configuration
        logger.info(f"Training {config.model_name.value} model")
        logger.info(f"Hyperparameter tuning: {config.fine_tuning}")
        
        # Set up MLflow logging
        if config.model_name == ModelType.LIGHTGBM:
            mlflow.lightgbm.autolog()
        elif config.model_name == ModelType.XGBOOST:
            mlflow.xgboost.autolog()
        else:
            mlflow.sklearn.autolog()
        
        # Hyperparameter tuning
        best_params = {}
        if config.fine_tuning:
            logger.info("Performing hyperparameter tuning...")
            best_params = tune_hyperparameters(
                config.model_name,
                x_train,
                y_train,
                n_trials=config.n_trials,
                cv=config.cv_folds,
                random_state=config.random_state
            )
            logger.info(f"Best parameters: {best_params}")
        
        # Train final model
        model = create_model(config.model_name, best_params)
        model.fit(x_train, y_train)
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': x_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info("\nFeature importance:")
            logger.info(feature_importance.to_string())
        
        return model
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise e