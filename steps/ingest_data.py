import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize the data ingestion class."""
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a file path.
    
    Args:
        data_path: Path to the input data file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {e}")
        raise

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: Path to the input data file
    Returns:
        df: pd.DataFrame
    """
    try:
        df = load_data(data_path)
        return df
    except Exception as e:
        logging.error(e)
        raise e