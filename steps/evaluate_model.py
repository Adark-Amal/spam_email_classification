from zenml import step
from src.models.model_evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import pandas as pd


@step
def evaluate_model(model: BaseEstimator, X_test: pd.Series, y_test: pd.Series):
    """
    Evaulates trained model and logs metrics
    
    Parameters:
        model
        X_test
        y_test
    
    Returns:
        None
    """
    
    me = ModelEvaluator()
    me.evaluate(model, X_test, y_test)

    