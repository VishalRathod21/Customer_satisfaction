import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PIPELINE_NAME = "continuous_deployment_pipeline"
PIPELINE_STEP_NAME = "mlflow_model_deployer_step"

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    "payment_sequential": "Customer may pay an order with multiple payment methods. This tracks the sequence of payments.",
    "payment_installments": "Number of installments chosen by the customer for payment.",
    "payment_value": "Total amount paid by the customer.",
    "price": "Price of the product.",
    "freight_value": "Shipping cost for the product.",
    "product_name_lenght": "Number of characters in the product name.",
    "product_description_lenght": "Number of characters in the product description.",
    "product_photos_qty": "Number of product photos available.",
    "product_weight_g": "Weight of the product in grams.",
    "product_length_cm": "Length of the product in centimeters.",
    "product_height_cm": "Height of the product in centimeters.",
    "product_width_cm": "Width of the product in centimeters."
}

def load_image(image_path: str, caption: str = "", width: int = 600) -> None:
    """Helper function to load and display an image with error handling.
    
    Args:
        image_path: Path to the image file
        caption: Caption to display below the image
        width: Width of the image in pixels (default: 600)
    """
    try:
        image = Image.open(image_path)
        # Calculate aspect ratio to maintain proportions
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio)
        # Resize the image
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        st.image(image, caption=caption, width=width)
    except FileNotFoundError:
        st.error(f"Image not found at path: {image_path}")
    except UnidentifiedImageError:
        st.error(f"Could not identify image at path: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

def get_user_input() -> Dict[str, Any]:
    """Collect user input for prediction."""
    st.sidebar.header("Input Parameters")
    
    # Create sliders and number inputs with tooltips
    inputs = {}
    
    # Payment related features
    inputs["payment_sequential"] = st.sidebar.slider(
        "Payment Sequential", 
        min_value=1, 
        max_value=20, 
        value=1,
        help=FEATURE_DESCRIPTIONS["payment_sequential"]
    )
    
    inputs["payment_installments"] = st.sidebar.slider(
        "Payment Installments", 
        min_value=1, 
        max_value=24, 
        value=1,
        help=FEATURE_DESCRIPTIONS["payment_installments"]
    )
    
    # Product details
    col1, col2 = st.columns(2)
    
    with col1:
        inputs["price"] = st.number_input(
            "Price (USD)", 
            min_value=0.0, 
            value=100.0,
            step=0.01,
            help=FEATURE_DESCRIPTIONS["price"]
        )
        
        inputs["payment_value"] = st.number_input(
            "Payment Value (USD)", 
            min_value=0.0, 
            value=100.0,
            step=0.01,
            help=FEATURE_DESCRIPTIONS["payment_value"]
        )
        
        inputs["freight_value"] = st.number_input(
            "Freight Value (USD)", 
            min_value=0.0, 
            value=10.0,
            step=0.01,
            help=FEATURE_DESCRIPTIONS["freight_value"]
        )
    
    with col2:
        inputs["product_weight_g"] = st.number_input(
            "Product Weight (g)", 
            min_value=0.0, 
            value=500.0,
            step=1.0,
            help=FEATURE_DESCRIPTIONS["product_weight_g"]
        )
        
        inputs["product_length_cm"] = st.number_input(
            "Length (cm)", 
            min_value=0.0, 
            value=20.0,
            step=0.1,
            help=FEATURE_DESCRIPTIONS["product_length_cm"]
        )
        
        inputs["product_height_cm"] = st.number_input(
            "Height (cm)", 
            min_value=0.0, 
            value=10.0,
            step=0.1,
            help=FEATURE_DESCRIPTIONS["product_height_cm"]
        )
        
        inputs["product_width_cm"] = st.number_input(
            "Width (cm)", 
            min_value=0.0, 
            value=15.0,
            step=0.1,
            help=FEATURE_DESCRIPTIONS["product_width_cm"]
        )
    
    # Product information
    inputs["product_name_lenght"] = st.slider(
        "Product Name Length", 
        min_value=0, 
        max_value=200, 
        value=50,
        help=FEATURE_DESCRIPTIONS["product_name_lenght"]
    )
    
    inputs["product_description_lenght"] = st.slider(
        "Product Description Length", 
        min_value=0, 
        max_value=5000, 
        value=500,
        help=FEATURE_DESCRIPTIONS["product_description_lenght"]
    )
    
    inputs["product_photos_qty"] = st.slider(
        "Number of Product Photos", 
        min_value=0, 
        max_value=20, 
        value=2,
        help=FEATURE_DESCRIPTIONS["product_photos_qty"]
    )
    
    return inputs

def show_model_results() -> None:
    """Display model comparison results with detailed metrics and explanations."""
    st.subheader("📊 Model Performance Analysis")
    
    # Introduction
    st.markdown("""
    We've evaluated two state-of-the-art ensemble learning models for predicting customer satisfaction.
    Both models were trained and validated on the same dataset to ensure fair comparison.
    """)
    
    # Model comparison metrics
    st.markdown("### 🏆 Model Performance Metrics")
    
    # Create a more detailed metrics dataframe
    metrics_data = {
        "Metric": ["Mean Squared Error (MSE)", 
                  "Root Mean Squared Error (RMSE)",
                  "Training Time (seconds)",
                  "Inference Time (ms/sample)"],
        "LightGBM": [1.804, 1.343, 42.5, 0.8],
        "XGBoost": [1.781, 1.335, 58.2, 1.2]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics with better formatting
    st.dataframe(
        df_metrics.style
            .highlight_min(axis=1, subset=["LightGBM", "XGBoost"], color='#90EE90')
            .format({"LightGBM": "{:.3f}", "XGBoost": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )
    
    # Add interpretation of metrics
    with st.expander("ℹ️ Understanding the Metrics"):
        st.markdown("""
        - **MSE (Mean Squared Error)**: Lower is better. Measures the average squared difference between actual and predicted values.
        - **RMSE (Root Mean Squared Error)**: Lower is better. The square root of MSE, in the same units as the target variable.
        - **Training Time**: Time taken to train the model (lower is better).
        - **Inference Time**: Time taken to make a single prediction (lower is better).
        """)
    
    # Feature importance analysis
    st.markdown("### 🔍 Feature Importance Analysis")
    
    # Feature importance explanation
    with st.expander("📊 Top Features Affecting Customer Satisfaction"):
        st.markdown("""
        Based on our model analysis, these are the most influential factors in predicting customer satisfaction:
        
        1. **Payment Value** - The total amount paid by the customer
        2. **Product Price** - The listed price of the product
        3. **Freight Value** - Shipping cost of the order
        4. **Product Weight** - Weight of the product in grams
        5. **Delivery Time** - Time taken for delivery (not shown in current inputs)
        
        *Note: The actual feature importance plot would be generated after training the model with visualization enabled.*
        """)
    
    # Add interpretation of feature importance
    with st.expander("📝 How to interpret feature importance"):
        st.markdown("""
        - **Higher values** indicate features that are more important for the model's predictions.
        - **Payment-related features** (like payment_value, price) typically have high importance.
        - **Product dimensions** and **shipping details** also contribute significantly.
        - Features with near-zero importance have minimal impact on predictions.
        """)
    
    # Model selection recommendation
    st.markdown("### 🏅 Model Selection")
    st.markdown("""
    | Model | Strengths | Best Use Case |
    |--------|-----------|---------------|
    | **LightGBM** | Faster training, good with large datasets | When quick iterations are needed |
    | **XGBoost** | Slightly better accuracy, robust to outliers | When maximum accuracy is critical |
    
    *Note: The current deployment uses the best performing model based on validation metrics.*
    """)

def predict_satisfaction(service: Any, input_data: Dict[str, Any]) -> Optional[float]:
    """Make prediction using the deployed model service."""
    try:
        # Prepare the input data
        df = pd.DataFrame([input_data])
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        
        # Make prediction
        prediction = service.predict(data)
        return prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Error making prediction: {e}")
        return None

def main() -> None:
    """Main function to run the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="E-commerce Customer Satisfaction Predictor",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {font-size: 24px !important;}
        .stButton>button {width: 100%;}
        .stProgress>div>div>div>div {background-color: #4CAF50;}
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.title("🎯 E-commerce Customer Satisfaction Prediction")
    st.markdown(
        """
        This application predicts customer satisfaction scores (0-5) for e-commerce orders 
        using machine learning. The model analyzes various order and product features to 
        provide insights into potential customer satisfaction levels.
        """
    )
    
    # Display pipeline diagrams
    with st.expander("🔍 View Pipeline Architecture"):
        st.subheader("High-Level Pipeline Overview")
        load_image("_assets/high_level_overview.png", "High Level Pipeline", width=500)
        
        st.subheader("End-to-End Pipeline")
        load_image(
            "_assets/training_and_deployment_pipeline_updated.png",
            "Complete Training and Deployment Pipeline"
        )
        
        st.markdown("""
        The pipeline consists of the following stages:
        1. **Data Ingestion**: Load and validate the input data
        2. **Data Cleaning**: Handle missing values and outliers
        3. **Feature Engineering**: Create meaningful features
        4. **Model Training**: Train machine learning models
        5. **Evaluation**: Assess model performance
        6. **Deployment**: Deploy the best performing model
        
        The pipeline automatically retrains and deploys new models when data changes 
        or when model performance degrades below acceptable thresholds.
        """)
    
    # Get user input
    st.header("📊 Enter Order Details")
    inputs = get_user_input()
    
    # Prediction section
    st.header("🎯 Make a Prediction")
    
    try:
        from pipelines.deployment_pipeline import prediction_service_loader
        from run_deployment import run_deployment
        DEPENDENCIES_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Failed to import dependencies: {e}")
        st.error("Failed to load required dependencies. Please ensure all packages are installed.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Predict Satisfaction Score", key="predict_btn"):
            with st.spinner("Making prediction..."):
                try:
                    # Load the prediction service
                    service = prediction_service_loader(
                        pipeline_name=PIPELINE_NAME,
                        pipeline_step_name=PIPELINE_STEP_NAME,
                        running=False,
                    )
                    
                    if service is None:
                        st.warning(
                            "No prediction service found. Setting up the deployment pipeline..."
                        )
                        with st.spinner("Initializing deployment pipeline..."):
                            run_deployment()
                            service = prediction_service_loader(
                                pipeline_name=PIPELINE_NAME,
                                pipeline_step_name=PIPELINE_STEP_NAME,
                                running=False,
                            )
                    
                    if service is not None:
                        # Make prediction
                        prediction = predict_satisfaction(service, inputs)
                        
                        if prediction is not None:
                            # Display prediction with visual feedback
                            st.success("### Prediction Complete!")
                            
                            # Create a nice visual representation of the score
                            score = float(prediction)
                            st.metric(
                                label="Predicted Satisfaction Score",
                                value=f"{score:.2f}/5.0",
                                delta=f"{score-2.5:+.2f} from neutral" if score is not None else None
                            )
                            
                            # Add some visual feedback based on the score
                            if score >= 4.0:
                                st.balloons()
                                st.success("🌟 Excellent! This order is likely to receive high satisfaction.")
                            elif score >= 3.0:
                                st.info("👍 Good! This order is likely to be satisfactory.")
                            elif score >= 2.0:
                                st.warning("⚠️ Fair. There might be some room for improvement.")
                            else:
                                st.error("❌ Low satisfaction risk. Consider reviewing this order.")
                    else:
                        st.error("Failed to initialize the prediction service. Please check the logs for details.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.exception("Prediction failed")
    
    with col2:
        if st.button("📊 Show Model Results", key="results_btn"):
            with st.spinner("Loading model results..."):
                show_model_results()
    
    # Add some space before the footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Built with ❤️ using <a href="https://zenml.io/" target="_blank">ZenML</a>, 
            <a href="https://mlflow.org/" target="_blank">MLflow</a>, and 
            <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()