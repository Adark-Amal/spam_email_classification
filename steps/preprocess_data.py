import pandas as pd
from zenml import step
import logging
from typing_extensions import Annotated
from typing import Tuple, Union
from src.data.preprocess import Preprocessor


@step
def preprocess_data(
    message_data: pd.DataFrame, 
    data_filepath: str
) -> Tuple[
    Annotated[Union[pd.DataFrame, pd.Series], "X_train"],
    Annotated[Union[pd.DataFrame, pd.Series], "X_test"],
    Annotated[Union[pd.DataFrame, pd.Series], "y_train"],
    Annotated[Union[pd.DataFrame, pd.Series], "y_test"]
]:
    """
    Preprocesses the provided dataframe, applying text preprocessing to each message.

    Parameters:
        message_data (pd.DataFrame): The original raw message data.
        data_filepath (str): The file path where the cleaned data will be saved.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
            - X_train: The training data features.
            - X_test: The test data features.
            - y_train: The training data labels.
            - y_test: The test data labels.
    """
    
    try:
        # Initialize preprocessor
        pmd = Preprocessor()
        
        # Preprocess the data
        cleaned_message_data = pmd.preprocess_dataframe(message_data)
        
        # Save the cleaned data to the specified file path
        cleaned_message_data.to_csv(data_filepath, index=False)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = pmd.split_data(cleaned_message_data)
        
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        return None, None, None, None