from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig, ModelType
from typing import Optional

@pipeline(enable_cache=True)
def training_pipeline(
    data_path: str,
    model_config: Optional[ModelNameConfig] = None
):
    """
    Training pipeline for the model.
    
    Args:
        data_path: Path to the input data file
        model_config: Configuration for the model to train. If None, uses default.
    """
    if model_config is None:
        model_config = ModelNameConfig()
        # Disable hyperparameter tuning for faster execution
        model_config.fine_tuning = False
        model_config.n_trials = 1
        model_config.cv_folds = 2
        
    # Force using LinearRegression model for faster training
    model_config.model_name = ModelType.LINEAR_REGRESSION

    # Ingest and prepare data
    data = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(data)
    
    # Train and evaluate model
    model = train_model(X_train, X_test, y_train, y_test, model_config)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
