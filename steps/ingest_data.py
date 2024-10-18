from zenml import step
import pandas as pd
import logging
from src.data.load_data import DataLoader

@step
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from the specified data path.

    Parameters:
        data_path (str): The file path where the data is located.

    Returns:
        pd.DataFrame: The loaded data as a Pandas DataFrame.
    """
    try:
        dl = DataLoader(data_path)
        message_data = dl.get_data()
        return message_data
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return None