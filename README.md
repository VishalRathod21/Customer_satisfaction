# E-Commerce Customer Satisfaction Predictor

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

## 🚀 Overview

This project is an advanced machine learning solution designed to forecast customer satisfaction levels for online purchases. By analyzing historical order data, this system helps businesses proactively address potential customer dissatisfaction and optimize their operations for better customer experiences.

## 🎯 Problem Statement

In the competitive e-commerce landscape, customer satisfaction is a critical factor that directly impacts business success. However, businesses often only become aware of customer dissatisfaction after receiving negative reviews or, worse, losing customers. The key challenges include:

1. **Reactive Approach**: Businesses typically respond to customer dissatisfaction after it occurs
2. **High Customer Churn**: Unhappy customers often don't complain but simply stop purchasing
3. **Operational Inefficiencies**: Without predictive insights, businesses can't proactively address potential issues
4. **Revenue Impact**: Poor customer experiences lead to decreased customer lifetime value and negative word-of-mouth

## 💡 Our Solution

This system addresses these challenges by:

1. **Predictive Analytics**: Leveraging machine learning to forecast customer satisfaction scores before orders are placed
2. **Proactive Intervention**: Enabling businesses to take preemptive actions for at-risk orders
3. **Data-Driven Decisions**: Providing actionable insights to improve various aspects of the customer journey
4. **Continuous Learning**: Automatically updating models with new data to improve prediction accuracy over time

The solution is built using MLOps best practices with ZenML, ensuring scalability, reproducibility, and easy deployment in production environments.

## 📊 Dataset

We use the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), which contains information on 100,000+ orders from 2016 to 2018 across multiple marketplaces in Brazil. The dataset includes:

- Order details (status, price, payment information)
- Customer information
- Product attributes
- Customer reviews and satisfaction scores

## 🛠️ Tech Stack

- **ML Framework**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **MLOps**: ZenML
- **Experiment Tracking**: MLflow
- **UI**: Streamlit
- **Hyperparameter Tuning**: Optuna

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-satisfaction.git
   cd customer-satisfaction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install ZenML and required integrations:
   ```bash
   pip install zenml["server"]
   zenml integration install mlflow -y
   ```

5. Initialize ZenML:
   ```bash
   zenml init
   ```

6. Set up the MLflow stack:
   ```bash
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml model-deployer register mlflow --flavor=mlflow
   zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
   ```

## 🏃‍♂️ Running the Pipeline

### Training Pipeline

To run the training pipeline:

```bash
python run_pipeline.py
```

### Deployment Pipeline

To run the continuous deployment pipeline:

```bash
python run_deployment.py
```

### Streamlit App

Run the Streamlit app to interact with the model:

```bash
streamlit run streamlit_app.py
```

## 📊 Model Performance

The pipeline supports multiple models with hyperparameter tuning:

- Linear Regression (default)
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Performance metrics (R² and RMSE) are tracked using MLflow.

## 📂 Project Structure

```
.
├── Data/                    # Dataset files
├── pipelines/               # ZenML pipeline definitions
│   ├── training_pipeline.py
│   ├── deployment_pipeline.py
│   └── utils.py
├── steps/                   # Pipeline steps
│   ├── clean_data.py
│   ├── evaluation.py
│   ├── ingest_data.py
│   └── model_train.py
├── src/                     # Source code
│   ├── data_cleaning.py
│   └── model_building.py
├── run_pipeline.py          # Script to run training
├── run_deployment.py        # Script to run deployment
├── streamlit_app.py         # Streamlit web interface
└── requirements.txt         # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
