import re
import pandas as pd
import nltk
import mlflow
from typing_extensions import Annotated
from typing import Tuple, Union
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')


class Preprocessor:

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

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
        text = re.sub(r"\W", " ", text)

        # Tokenize sentence
        tokens = nltk.word_tokenize(text)

        # Remove stop words and apply stemming
        tokens = [
            self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words
        ]

        return " ".join(tokens)

    def preprocess_dataframe(self, message_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess an entire dataframe, applying text preprocessing to each message.

        Parameters:
            message_data (pd.DataFrame): DataFrame containing the dataset with a 'message' column.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        
        # Apply text normalization to message column and encode labels
        message_data["cleaned_message"] = message_data["message"].apply(
            self.preprocess_text
        )
        message_data.drop(columns=["message"], inplace=True)
        message_data["label"] = message_data.label.map({"ham": 0, "spam": 1})

        # Log preprocessing settings to MLflow
        with mlflow.start_run(nested=True, run_name="Preprocessing Data"):
            mlflow.log_param("preprocessing_stopwords", len(self.stop_words))
            mlflow.log_param("preprocessing_lemmatizer", type(self.lemmatizer).__name__)
            mlflow.log_param("num_rows_in_dataframe", len(message_data))
            mlflow.log_param("encoded_labels", {"ham": 0, "spam": 1})

        return message_data


    def split_data(self, message_data: pd.DataFrame) -> Tuple[
        Annotated[Union[pd.DataFrame, pd.Series], "X_train"],
        Annotated[Union[pd.DataFrame, pd.Series], "X_test"],
        Annotated[Union[pd.DataFrame, pd.Series], "y_train"],
        Annotated[Union[pd.DataFrame, pd.Series], "y_test"]
    ]:
        """
        Split data into train and test
        
        Parameters:
            message_data (pd.DataFrame): DataFrame containing the dataset with a 'message' column.
        
        Returns:
            X_train (pd.Series): Sample data for training model
            X_test (pd.Series): Sample data for evaluating model
            y_train (pd.Series): Sample data for training model
            y_test (pd.Series): Sample data for evaluating model
        """
        
        with mlflow.start_run(nested=True, run_name="Splitting Data"):
            X_train, X_test, y_train, y_test = train_test_split(
                    message_data["cleaned_message"],
                    message_data["label"],
                    test_size=0.2,
                    random_state=2024,
                )

            mlflow.log_param("train_data_size", len(X_train))
            mlflow.log_param("test_data_size", len(X_test))
            
        return X_train, X_test, y_train, y_test