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

## 🐳 Docker Deployment

You can easily deploy this application using Docker. Here's how:

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your system

### Building the Docker Image

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/VishalRathod21/Customer_satisfaction.git
   cd Customer_satisfaction
   ```

2. Build the Docker image:
   ```bash
   docker build -t customer-satisfaction .
   ```

### Running the Container

1. Run the container with port mapping (maps container port 8501 to host port 8501):
   ```bash
   docker run -p 8501:8501 customer-satisfaction
   ```

2. Access the application in your browser at:
   ```
   http://localhost:8501
   ```

### Environment Variables

You can configure the application using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to run the Streamlit app | `8501` |
| `ZENML_MLFLOW_TRACKING_URI` | MLflow tracking server URI | `http://localhost:5000` |

Example with environment variables:
```bash
docker run -p 8501:8501 \
  -e PORT=8501 \
  -e ZENML_MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 \
  customer-satisfaction
```

### Docker Compose

For a more complex setup with MLflow tracking, you can use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PORT=8501
      - ZENML_MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    volumes:
      - ./_assets:/app/_assets

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns
    volumes:
      - mlflow_data:/mlruns

volumes:
  mlflow_data:
```

To start with Docker Compose:
```bash
docker-compose up -d
```

### Building for Production

For production deployment, you might want to use a multi-stage build. Here's an enhanced `Dockerfile`:

```dockerfile
# Build stage
FROM python:3.10.13-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10.13-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Monitoring and Logging

For production deployments, consider adding:

1. **Logging**: Configure proper logging to STDOUT/STDERR
2. **Health Checks**: Add a health check endpoint
3. **Metrics**: Expose Prometheus metrics for monitoring
4. **Tracing**: Add distributed tracing for better observability

### CI/CD Integration

You can integrate this with your CI/CD pipeline to automatically build and deploy new versions of your application.

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
