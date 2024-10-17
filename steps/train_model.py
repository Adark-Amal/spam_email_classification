from zenml import step
import pandas as pd
from src.models.model_training import ModelTrainer
from sklearn.base import BaseEstimator


@step
def train_model(X_train: pd.Series, y_train: pd.Series) -> BaseEstimator:
    """
    Run the model training pipeline
    
    Parameters:
        X_train
        y_train
    
    Returns:
        trained model
    """
    
    tm = ModelTrainer()
    model = tm.train(X_train, y_train)
    
    return model

@step
def save_model(model: BaseEstimator, file_path: str) -> None:
    """
    Save best model
    
    Parameters:
        X_train
        y_train
    
    Returns:
        None
    """
    tm = ModelTrainer()
    tm.write_model(model, file_path)