import logging 
import pandas as pd
from zenml import step

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IngestData:
    """Ingest data from a CSV file."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        """Ingest data from a CSV file."""
        logger.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file."""
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise e