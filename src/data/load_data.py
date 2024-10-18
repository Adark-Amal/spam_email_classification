import pandas as pd
import mlflow

class DataLoader:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def get_data(self) -> pd.DataFrame:
        """
        Loads the raw data from the specified filepath.

        Parameters:
            None

        Returns:
            pd.DataFrame: Loaded dataset as a pandas DataFrame.
        """
        # Load the data
        message_data = pd.read_table(
            self.filepath, sep="\t", header=None, names=["label", "message"]
        )
        label_counts = message_data["label"].value_counts().to_dict()
        
        # Log details about the dataset using MLflow
        with mlflow.start_run(nested=True, run_name="Data Loading"):
            mlflow.log_param("data_filepath", self.filepath)
            mlflow.log_param("num_samples", len(message_data))
            mlflow.log_param("num_features", len(message_data.columns))
            mlflow.log_param("features", list(message_data.columns))
            mlflow.log_param("label_distribution", label_counts)

        return message_data

