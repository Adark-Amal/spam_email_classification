import pandas as pd
import mlflow


class DataLoader:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw data from the specified filepath and logs details with MLflow.

        Parameters:
            None

        Returns:
            pd.DataFrame: Loaded dataset as a pandas DataFrame.
        """
        with mlflow.start_run(run_name="Data Loading"):
            try:
                # Load the data
                message_data = pd.read_table(
                    self.filepath, sep="\t", header=None, names=["label", "message"]
                )
                label_counts = message_data["label"].value_counts().to_dict()

                # Log details about the dataset using MLflow
                mlflow.log_param("data_filepath", self.filepath)
                mlflow.log_param("num_samples", len(message_data))
                mlflow.log_param("num_features", len(message_data.columns))
                mlflow.log_param("label_distribution", label_counts)

                return message_data
            except FileNotFoundError as e:
                mlflow.log_param("data_loading_error", str(e))
                print(f"Error: {e}")
                return None
