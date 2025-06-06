from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, List, Optional

class ModelType(str, Enum):
    LINEAR_REGRESSION = "LinearRegression"
    RANDOM_FOREST = "RandomForest"
    LIGHTGBM = "LightGBM"
    XGBOOST = "XGBoost"
    GRADIENT_BOOSTING = "GradientBoosting"

class ModelNameConfig(BaseModel):
    model_name: ModelType = Field(
        default=ModelType.LIGHTGBM,  # Faster model by default
        description="The type of model to train"
    )
    fine_tuning: bool = Field(
        default=False,
        description="Whether to perform hyperparameter tuning. Set to False for faster training."
    )
    cv_folds: int = Field(
        default=3,
        description="Number of cross-validation folds for hyperparameter tuning. Reduced for faster training."
    )
    n_trials: int = Field(
        default=10,
        description="Number of trials for hyperparameter optimization. Reduced for faster training."
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )

# Hyperparameter search spaces for each model type (optimized for speed)
MODEL_PARAMS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.RANDOM_FOREST: {
        "n_estimators": [100],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True],
        "n_jobs": [-1]  # Use all available cores
    },
    ModelType.XGBOOST: {
        "n_estimators": [100],
        "max_depth": [3, 6],
        "learning_rate": [0.1],
        "n_jobs": [-1]  # Use all available cores
    },
    ModelType.LIGHTGBM: {
        "n_estimators": [100],
        "max_depth": [5],
        "learning_rate": [0.1],
        "num_leaves": [31],
        "n_jobs": [-1],  # Use all available cores
        "verbose": [-1]  # Reduce output verbosity
    },
    ModelType.GRADIENT_BOOSTING: {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    },
    ModelType.LINEAR_REGRESSION: {
        "fit_intercept": [True],
        "n_jobs": [-1]  # Use all available cores
    }
}
