import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple, Union
from src.data.preprocess import Preprocessor


@step
def preprocess_data(message_data: pd.DataFrame, data_filepath: str) -> Tuple[
        Annotated[Union[pd.DataFrame, pd.Series], "X_train"],
        Annotated[Union[pd.DataFrame, pd.Series], "X_test"],
        Annotated[Union[pd.DataFrame, pd.Series], "y_train"],
        Annotated[Union[pd.DataFrame, pd.Series], "y_test"]
    ]:
        """
        Preprocess an entire dataframe, applying text preprocessing to each message.

        Parameters:
            message_data (pd.DataFrame): Original data

        Returns:
            cleaned_message_data
            X_train
            X_test
            y_train
            y_test
        """
        
        try:
            pmd = Preprocessor()
            cleaned_message_data = pmd.preprocess_dataframe(message_data)
            cleaned_message_data.to_csv(data_filepath, index=False)
            X_train, X_test, y_train, y_test = pmd.split_data(cleaned_message_data)
            #print(len(X_train, len(y_train)))
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(e)
