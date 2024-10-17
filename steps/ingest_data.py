from zenml import step
import pandas as pd
from src.data.load_data import DataLoader

@step
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from defined data path
    
    Parameters:
        data_path (str): Path to data
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        dl = DataLoader(data_path)
        message_data = dl.get_data()
        return message_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None