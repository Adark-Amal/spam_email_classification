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
                df = pd.read_table(self.filepath, sep='\t', header=None, names=['label', 'message'])

                # Log details about the dataset using MLflow
                mlflow.log_param('data_filepath', self.filepath)  
                mlflow.log_param('num_samples', len(df))  
                mlflow.log_param('num_features', len(df.columns))  

                # Log the label distribution (optional but useful for class imbalance analysis)
                label_counts = df['label'].value_counts().to_dict()
                mlflow.log_param('label_distribution', label_counts)

                return df

            except FileNotFoundError as e:
                # Log an error if the file is not found
                mlflow.log_param('data_loading_error', str(e))  
                print(f"Error: {e}")
                return None
