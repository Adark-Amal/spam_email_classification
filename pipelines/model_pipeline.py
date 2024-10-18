from steps.ingest_data import load_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from zenml import pipeline
import logging
import mlflow

# Set the experiment name
mlflow.set_experiment("Spam Classifier")


@pipeline(name="Spam Classifier", enable_cache=False)
def model_pipeline(data_path: str):
    """
    A pipeline for loading data, preprocessing it, training a model, saving the model, and evaluating it.

    Parameters:
        data_path (str): The file path where the raw data is located.

    Returns:
        None
    """
    model_filepath: str = "models/spam_classifier_pipeline.pkl"
    data_filepath: str = "data/processed_data/processed_sms_data.csv"

    try:
        # Load data
        message_data = load_data(data_path)

        # Preprocess and save data
        X_train, X_test, y_train, y_test = preprocess_data(message_data, data_filepath)

        # Train and save model
        model = train_model(X_train, y_train, model_filepath)

        # Evaluate Model
        evaluate_model(model, X_train, y_train, X_test, y_test)
    except Exception as e:
        logging.error(f"An error occurred in the model pipeline: {e}")
