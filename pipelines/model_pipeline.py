from steps.ingest_data import load_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model, save_model
from steps.evaluate_model import evaluate_model
from zenml import pipeline


@pipeline(enable_cache=False)
def model_pipeline(data_path: str):
    
    model_filepath: str = "models/spam_classifier_pipeline.pkl"
    data_filepath: str = "data/processed_data/processed_sms_data.csv"
    
    # Load data
    message_data = load_data(data_path)
    
    # Preprocess and save data
    X_train, X_test, y_train, y_test = preprocess_data(message_data, data_filepath)
    
    # Train and save model
    model = train_model(X_train, y_train)
    save_model(model, model_filepath)
    
    # Evaluate Model
    evaluate_model(model, X_test, y_test)


