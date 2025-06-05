import logging 
import pandas as pd
from zenml import step

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IngestData:
    """Ingest data from a CSV file."""
    def __init__(self)->None:
        pass

    def get_data(self)->pd.DataFrame:
        """Ingest data from a CSV file."""
        
        return pd.read_csv("/home/vishalr/Desktop/Mlops project/Data/olist_customers_dataset.csv")
    
@step
def ingest_df() -> pd.DataFrame:
    """Ingest data from a CSV file."""
    try:
        ingest_data = IngestData()
        return ingest_data.get_data()
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise e