import pytest
from src.data.load_data import DataLoader
from src.models.train_model import ModelTrainer
from sklearn.metrics import accuracy_score

# Test model training
def test_model_training():
    """
    Test the model training process to ensure that the model is properly trained,
    and a valid test set is created.
    """
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()

    trainer = ModelTrainer()
    trainer.train(raw_data)
    
    # Check that the model is trained and the pipeline is initialized
    assert trainer.model is not None, "Model training failed, pipeline not initialized"
    
    # Ensure that a test set was created and it's not empty
    assert len(trainer.X_test) > 0, "Test set is empty after train-test split"
    assert len(trainer.y_test) > 0, "Test set labels are empty after train-test split"

# Test model prediction
def test_model_prediction():
    """
    Test the model's prediction capabilities by ensuring that predictions can
    be made on the test set and that they match the size of the test set.
    """
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()

    trainer = ModelTrainer()
    trainer.train(raw_data)

    # Make predictions on the test set
    y_pred = trainer.model.predict(trainer.X_test)
    
    # Check that the number of predictions matches the number of test labels
    assert len(y_pred) == len(trainer.y_test), "Prediction length mismatch with test set"
    
    # Optionally: Ensure the model is making non-empty predictions
    assert y_pred is not None, "Model predictions failed, no output"
    
    # Optionally: Check that the predictions contain valid values (0 or 1 for ham/spam)
    assert set(y_pred).issubset({0, 1}), "Model prediction values are outside the expected range (0 for ham, 1 for spam)"

# Test model accuracy
def test_model_accuracy():
    """
    Test the accuracy of the trained model by calculating accuracy on the test set.
    """
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()

    trainer = ModelTrainer()
    trainer.train(raw_data)

    # Make predictions on the test set
    y_pred = trainer.model.predict(trainer.X_test)

    # Calculate accuracy
    accuracy = accuracy_score(trainer.y_test, y_pred)
    
    # Ensure that the accuracy is within a reasonable range
    assert accuracy > 0.8, f"Model accuracy is too low: {accuracy}"
