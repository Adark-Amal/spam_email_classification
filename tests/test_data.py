import pytest
from src.data.load_data import DataLoader
from src.data.preprocess import Preprocessor

# Test loading of raw data
def test_load_data():
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    df = data_loader.load_data()
    
    # Check if data was loaded successfully
    assert df is not None, "Data loading failed"
    
    # Check if required columns are present
    assert 'label' in df.columns, "Data missing 'label' column"
    assert 'message' in df.columns, "Data missing 'message' column"
    
    # Ensure data isn't empty
    assert len(df) > 0, "Loaded data is empty"

    # Check if data contains the expected labels ('ham', 'spam')
    assert set(df['label'].unique()) == {'ham', 'spam'}, "Unexpected labels in data"

# Test preprocessing of data
def test_preprocess_data():
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()
    preprocessor = Preprocessor()
    
    # Preprocess the data
    processed_data = preprocessor.preprocess_dataframe(raw_data)
    
    # Ensure the preprocessed data has the required 'cleaned_message' column
    assert 'cleaned_message' in processed_data.columns, "Preprocessing failed, 'cleaned_message' column missing"
    
    # Ensure the cleaned message column is not empty
    assert processed_data['cleaned_message'].isnull().sum() == 0, "Preprocessed data contains empty cleaned messages"

# Test preprocessing with an empty dataframe
def test_preprocess_empty_dataframe():
    preprocessor = Preprocessor()
    
    # Create an empty dataframe
    empty_df = DataLoader(filepath='data/raw_data/EmptyFile').load_data()
    
    # Preprocess the empty dataframe
    processed_data = preprocessor.preprocess_dataframe(empty_df)
    
    # Ensure the result is also an empty dataframe
    assert processed_data.empty, "Preprocessing on an empty dataframe should return an empty dataframe"

# Test handling of special characters in messages
def test_preprocess_special_characters():
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()
    
    # Add special characters and numbers to the data
    raw_data.loc[0, 'message'] = "!!Hello World$$@@ 123"
    
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess_dataframe(raw_data)
    
    # Check that the message has been cleaned correctly (special characters removed)
    assert processed_data['cleaned_message'].iloc[0] == "hello world 123", "Preprocessing failed to clean special characters"
