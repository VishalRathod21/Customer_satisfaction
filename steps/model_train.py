import logging
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml import step
from .config import ModelNameConfig
from zenml.client import Client
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a Linear Regression model with or without hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Training features
        x_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        config (ModelNameConfig): Configuration for training

    Returns:
        Trained RegressorMixin model
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model =model.train(x_train, y_train)
            logging.info("Model training complete")
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not found")

    except Exception as e:
        logging.error(e)
        raise e
