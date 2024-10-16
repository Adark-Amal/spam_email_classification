import re
import pandas as pd
import mlflow
import nltk
from typing import List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

class Preprocessor:

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text message (lowercase, remove punctuation, tokenize, remove stop words, and stem).
        
        Parameters:
            text (str): Raw text message to preprocess.
            
        Returns:
            str: Preprocessed text message.
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        
        # Tokenize sentence
        tokens = nltk.word_tokenize(text)
        
        # Remove stop words and apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        # Join tokens back into a single string
        return ' '.join(tokens)
    
    def preprocess_text_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocesses a batch of texts by applying `self.preprocess_text` to each one.
        
        Args:
            texts (list of str): List of raw text data.
        
        Returns:
            list of str: Preprocessed text data.
        """
        # Check if the parameter 'num_texts_to_preprocess' has already been logged
        if mlflow.active_run():
            current_run = mlflow.get_run(mlflow.active_run().info.run_id)
            existing_param = current_run.data.params.get('num_texts_to_preprocess')

            # Log the parameter only if it hasn't been logged already
            if existing_param is None:
                mlflow.log_param('num_texts_to_preprocess', len(texts))
        
        # Preprocess the texts
        return [self.preprocess_text(text) for text in texts]
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess an entire dataframe, applying text preprocessing to each message.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the dataset with a 'message' column.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'cleaned_message' column containing preprocessed text.
        """
        
        df['cleaned_message'] = df['message'].apply(self.preprocess_text)
        
        if mlflow.active_run():
            mlflow.end_run()
            
         # Log the final model with a new run specifically for model logging
        with mlflow.start_run(run_name="Data Preprocessing"):
            # Log preprocessing settings to MLflow
            mlflow.log_param('preprocessing_stopwords', len(self.stop_words))  
            mlflow.log_param('preprocessing_stemmer', type(self.stemmer).__name__) 
            mlflow.log_param('num_rows_in_dataframe', len(df))
        
        return df
