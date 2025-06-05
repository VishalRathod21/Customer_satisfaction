from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    """
    Training pipeline for the model.
    
    Args:
        data_path: Path to the input data file
    """
    # Ingest and prepare data
    data = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(data)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    r2_score, rmse = evaluate_model(model, X_test, y_test)




    