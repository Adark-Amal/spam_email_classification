from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.data.preprocess import *
import joblib
import numpy as np
import mlflow

class FeatureBuilder(Preprocessor):

    def __init__(self):
        super().__init__()
        # Initialize the pipeline and log parameters
        self.pipeline = Pipeline([
            ('preprocess', FunctionTransformer(self.preprocess_text_batch, validate=False)),
            ('tfidf', TfidfVectorizer())  
        ])
            
        # Log the parameters for TF-IDF
        vectorizer_params = self.pipeline.named_steps['tfidf'].get_params()

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits the pipeline on the training data and transforms it. Logs relevant parameters and the shape of the feature matrix.

        Parameters:
            df (pd.DataFrame): DataFrame containing the raw text data in the 'message' column.
            
        Returns:
            sparse matrix: TF-IDF feature matrix after applying preprocessing and TF-IDF.
        """
        # Fit and transform the data
        feature_matrix = self.pipeline.fit_transform(df['message'])

        return feature_matrix

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data using the already fitted pipeline.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the raw text data in the 'message' column.
            
        Returns:
            sparse matrix: TF-IDF feature matrix after applying preprocessing and TF-IDF.
        """
        feature_matrix = self.pipeline.transform(df['message'])

        return feature_matrix

    def save_pipeline(self, filepath: str ='models/tfidf_pipeline.pkl') -> None:
        """
        Saves the entire pipeline (preprocessing + TF-IDF vectorizer) to a file and logs it as an MLflow artifact.
        
        Parameters:
            filepath (str): Path where the pipeline should be saved.
        
        Returns:
            None
        """
        joblib.dump(self.pipeline, filepath)
        print(f"TF-IDF pipeline saved at {filepath}")

    def load_pipeline(self, filepath: str ='models/tfidf_pipeline.pkl') -> None:
        """
        Loads a previously saved pipeline (preprocessing + TF-IDF) from a file.
        
        Parameters:
            filepath (str): Path to the saved pipeline file.
        """
        self.pipeline = joblib.load(filepath)
        print(f"TF-IDF pipeline loaded from {filepath}")