import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreProcessingStrategy,
)


@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Clean data from a CSV file."""
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e