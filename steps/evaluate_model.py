from zenml import step
from src.models.model_evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import pandas as pd
import logging


@step
def evaluate_model(
    model: BaseEstimator,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Evaluates the trained model on the test data and logs the metrics.

    Parameters:
        model (BaseEstimator): The trained model to be evaluated.
        X_train (pd.Series): The training data features.
        y_train (pd.Series): The training data labels.
        X_test (pd.Series): The test data features.
        y_test (pd.Series): The test data labels.

    Returns:
        None
    """

    try:
        me = ModelEvaluator()
        me.evaluate(model, X_train, y_train, X_test, y_test)
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
