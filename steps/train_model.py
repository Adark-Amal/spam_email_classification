from zenml import step
import pandas as pd
from src.models.model_training import ModelTrainer
from sklearn.base import BaseEstimator
import logging
from typing import Union
import mlflow


@step
def train_model(X_train: pd.Series, y_train: pd.Series, file_path: str) -> Union[BaseEstimator, None]:
    """
    Runs the model training pipeline using the provided training data.

    Parameters:
        X_train (pd.Series): The training data features.
        y_train (pd.Series): The training data labels.
        file_path (str): The file path where the model will be saved.

    Returns:
        Union[BaseEstimator, None]: The trained model if successful, otherwise returns None.
    """
    
    try:
        # Log the final model with the signature and input example
        with mlflow.start_run(run_name="Train Model"):
            tm = ModelTrainer()
            model = tm.train(X_train, y_train)
            tm.write_model(model, file_path)
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None
