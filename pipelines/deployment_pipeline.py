import json
import logging
from typing import Tuple
import numpy as np 
import pandas as pd
from pydantic import BaseModel
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.config import ModelNameConfig, ModelType
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Docker settings for the pipeline
docker_settings = DockerSettings(
    required_integrations=[MLFLOW]
)

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Loads a small sample of test data for inference."""
    # Load sample order and product data for prediction
    import pandas as pd
    
    # Create sample data with the expected features
    sample_data = [
        {
            "payment_sequential": 1,
            "payment_installments": 1,
            "payment_value": 100.0,
            "price": 100.0,
            "freight_value": 10.0,
            "product_name_lenght": 50,
            "product_description_lenght": 500,
            "product_photos_qty": 2,
            "product_weight_g": 500,
            "product_length_cm": 20,
            "product_height_cm": 10,
            "product_width_cm": 15
        }
    ]
    
    # Convert to DataFrame and then to JSON
    df = pd.DataFrame(sample_data)
    print("Sample data for prediction:")
    print(df)
    
    return df.to_json(orient="records")


class DeploymentTriggerConfig(BaseModel):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.0


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    try:
        print("Starting service...")
        service.start(timeout=10)  # should be a NOP if already started
        
        # Parse the JSON string into a list of dictionaries
        print("Parsing JSON data...")
        records = json.loads(data)
        print(f"Loaded {len(records)} records for prediction")
        
        # Convert the list of dictionaries to a DataFrame
        print("Creating DataFrame...")
        df = pd.DataFrame(records)
        print(f"DataFrame shape: {df.shape}")
        print("Available columns:", df.columns.tolist())
        
        # Select only the required columns
        required_columns = [
            "payment_sequential",
            "payment_installments",
            "payment_value",
            "price",
            "freight_value",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
        
        print("Required columns:", required_columns)
        
        # Ensure all required columns exist in the DataFrame
        for col in required_columns:
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = 0  # Fill missing columns with 0
        
        # Select and reorder columns
        df = df[required_columns]
        
        print("DataFrame after column selection:")
        print(df.head())
        
        # Convert to the format expected by the model
        data_array = df.values
        print(f"Data array shape: {data_array.shape}")
        print(f"Data array type: {type(data_array)}")
        print(f"Data array dtype: {data_array.dtype}")
        
        # The MLflow deployment service expects a numpy array
        print("Ensuring data is in the correct format for MLflow...")
        
        # Make sure the data is a numpy array
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)
            
        print(f"Data array shape: {data_array.shape}")
        print(f"First row: {data_array[0]}")
        
        print("Sending data to model for prediction...")
        
        try:
            # Try to make the prediction with the numpy array directly
            prediction = service.predict(data_array)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print("Trying alternative data format...")
            
            # If that fails, try converting to a DataFrame and then to records
            column_names = [
                "payment_sequential", "payment_installments", "payment_value",
                "price", "freight_value", "product_name_lenght",
                "product_description_lenght", "product_photos_qty",
                "product_weight_g", "product_length_cm", "product_height_cm",
                "product_width_cm"
            ]
            df = pd.DataFrame(data_array, columns=column_names)
            prediction = service.predict(df)
        
        print("Prediction completed successfully!")
        if hasattr(prediction, 'shape'):
            print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction values: {prediction}")
        
        return prediction
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw data: {data}")
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response from server: {e.response.text}")
        raise


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    
    # Initialize model config with hyperparameter tuning disabled for faster execution
    model_config = ModelNameConfig(
        model_name=ModelType.RANDOM_FOREST,
        fine_tuning=False,  # Disable hyperparameter tuning
        n_trials=1,  # Minimum trials
        cv_folds=2  # Minimum cross-validation folds
    )
    
    # Train the model
    model = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        config=model_config
    )
    
    # Evaluate the model
    r2_score, rmse = evaluate_model(model, x_test, y_test)
    
    # Create deployment trigger config
    trigger_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    
    # Make deployment decision
    deployment_decision = deployment_trigger(
        accuracy=r2_score,  # Using r2_score as the accuracy metric
        config=trigger_config
    )
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


from zenml.config import DockerSettings

# Configure MLflow tracking
mlflow_settings = {
    "experiment_name": "inference_pipeline",
    "tracking_uri": "file:///home/vishalr/.config/zenml/local_stores/de11fa3a-6687-44d9-a2b9-fd074f29b15c/mlruns"
}

@pipeline(enable_cache=False, settings={
    "docker": docker_settings,
    "experiment_tracker": {
        "experiment_name": "inference_pipeline",
        "tracking_uri": "file:///home/vishalr/.config/zenml/local_stores/de11fa3a-6687-44d9-a2b9-fd074f29b15c/mlruns"
    }
})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Import MLflow and ZenML client
    import mlflow
    from zenml.client import Client
    
    # Get the MLflow experiment tracker
    experiment_tracker = Client().active_stack.experiment_tracker
    
    # Set the MLflow tracking URI and experiment
    tracking_uri = "file:///home/vishalr/.config/zenml/local_stores/de11fa3a-6687-44d9-a2b9-fd074f29b15c/mlruns"
    print(f"Using tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("inference_pipeline")
    
    # Start an MLflow run
    with mlflow.start_run(run_name="inference_run") as run:
        print(f"MLflow run started with ID: {run.info.run_id}")
        # Log a simple metric to verify MLflow is working
        mlflow.log_metric("test_metric", 1.0)
        print("Logged test metric to MLflow")
        
        # Run the prediction
        batch_data = dynamic_importer()
        model_deployment_service = prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            running=True,
        )
        prediction = predictor(service=model_deployment_service, data=batch_data)
        
        print("Inference completed successfully!")
        print(f"Raw prediction result: {prediction}")
        
        # Log the prediction result
        try:
            # Try to log the prediction value if possible
            if hasattr(prediction, 'item'):
                prediction_value = float(prediction.item())
                mlflow.log_metric("prediction", prediction_value)
                print(f"Logged prediction value: {prediction_value}")
            else:
                print("Could not extract prediction value for MLflow logging")
        except Exception as e:
            print(f"Error logging prediction to MLflow: {e}")
        
        return prediction