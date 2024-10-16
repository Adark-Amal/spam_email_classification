import joblib

def load_model(model_path: str) -> object:
    """
    Load the trained model pipeline (which includes the vectorizer and classifier).
    
    Parameters:
        model_path (str): Path to the saved model pipeline.
    
    Returns:
        Pipeline: The loaded model pipeline.
    """
    return joblib.load(model_path)

def predict_spam(model: object, message: str) -> str:
    """
    Use the loaded model pipeline to predict if a message is spam or ham.
    
    Parameters:
        model (Pipeline): The loaded model pipeline.
        message (str): The message to classify.
    
    Returns:
        str: 'Spam' if the message is classified as spam, 'Ham' otherwise.
    """
    prediction = model.predict([message])
    return 'Spam' if prediction[0] == 1 else 'Ham'

def test_model():
    # Path to the saved model (pipeline)
    model_path = 'models/spam_classifier_pipeline.pkl'
    
    # Load the model
    print("Loading the model...")
    model = load_model(model_path)
    
    # Example message for prediction
    test_message = "Congratulations! You've won a free iPhone. Claim now."
    
    # Make a prediction
    result = predict_spam(model, test_message)
    
    # Display the result
    print(f"Message: {test_message}")
    print(f"Prediction: {result}")

if __name__ == "__main__":
    test_model()
