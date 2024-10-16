import pytest
from src.data.load_data import DataLoader
from src.features.build_features import FeatureBuilder

# Test feature extraction with TF-IDF
def test_feature_extraction():
    """
    Test the feature extraction process using TF-IDF and ensure that the generated
    matrix has the correct shape and contains features.
    """
    data_loader = DataLoader(filepath='data/raw_data/SMSSpamCollection')
    raw_data = data_loader.load_data()
    
    feature_builder = FeatureBuilder()
    tfidf_matrix = feature_builder.fit_transform(raw_data)
    
    # Check if the number of rows in the TF-IDF matrix matches the number of raw data samples
    assert tfidf_matrix.shape[0] == len(raw_data), "TF-IDF feature extraction failed: Incorrect number of samples"
    
    # Check if the number of features (columns) in the TF-IDF matrix is greater than 0
    assert tfidf_matrix.shape[1] > 0, "TF-IDF feature extraction failed: No features in the TF-IDF matrix"

def test_feature_extraction_special_characters():
    """
    Test the feature extraction process when special characters are present in the text
    and ensure that the special characters are handled correctly.
    """
    # Simulate a dataframe with special characters
    raw_data = DataLoader(filepath='data/raw_data/SMSSpamCollection').load_data()
    
    # Modify the raw data to include special characters
    raw_data.loc[0, 'message'] = "Hello!!! How are you??? $$$ 123"
    
    feature_builder = FeatureBuilder()
    tfidf_matrix = feature_builder.fit_transform(raw_data)
    
    # Ensure the special characters are handled properly (TF-IDF should work without errors)
    assert tfidf_matrix.shape[0] == len(raw_data), "TF-IDF feature extraction failed: Incorrect handling of special characters"
