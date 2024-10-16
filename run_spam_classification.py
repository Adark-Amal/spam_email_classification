import os
from src.data.load_data import *
from src.data.preprocess import *
from src.features.build_features import *
from src.models.train_model import *
from src.models.evaluate_model import *
from src.visualization.visualize import *


def run_project():
    """
    The main function to execute the complete spam classification workflow:

    Parameters:
        None

    Returns:
        None
    """
    # Create necessary directories if they don't exist
    os.makedirs("data/processed_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    data_loader = DataLoader(filepath="data/raw_data/SMSSpamCollection")
    raw_data = data_loader.load_data()

    print("Preprocessing data...")
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess_dataframe(raw_data)

    processed_data.to_csv("data/processed_data/processed_sms_data.csv", index=False)
    print("Processed data saved to 'data/processed/processed_sms_data.csv'.")

    print("Training model with pipeline...")
    trainer = ModelTrainer()
    trainer.train(raw_data)

    trainer.save_model(filepath="models/spam_classifier_pipeline.pkl")
    print("Trained model (pipeline) saved to 'models/spam_classifier_pipeline.pkl'.")

    print("Evaluating model...")
    evaluator = ModelEvaluator(trainer.model, trainer.X_test, trainer.y_test)
    evaluator.evaluate()


def load_model(model_path: object) -> object:
    """
    Load the trained model pipeline from the specified path.

    Args:
        model_path (str): Path to the saved model pipeline.

    Returns:
        Pipeline: The loaded model pipeline.
    """
    return joblib.load(model_path)


if __name__ == "__main__":
    run_project()
