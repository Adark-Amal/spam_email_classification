import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from src.data.preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ModelTrainer(Preprocessor):

    def __init__(self):
        super().__init__()

    def train(self, df: pd.DataFrame) -> None:
        """
        Trains the model using RandomizedSearchCV followed by GridSearchCV.
        Logs the metrics (accuracy, F1 score, precision, recall) over each fold.
        Logs model signatures, artifacts, and custom hyperparameters.
        """
        
        df['label'] = df.label.map({'ham': 0, 'spam': 1})
        
        X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=2024)

 
        pipeline = Pipeline([
                ('preprocess', FunctionTransformer(self.preprocess_text_batch, validate=False)),
                ('tfidf', TfidfVectorizer()),  
                ('classifier', RandomForestClassifier(random_state=2024))  
            ])

        param_distributions = [
            {
                'classifier': [RandomForestClassifier(random_state=2024)],
                'classifier__n_estimators': np.arange(50, 200, 50),
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': np.arange(2, 10),
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            {
                'classifier': [LogisticRegression(random_state=2024)],
                'classifier__C': np.logspace(-3, 3, 7),
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            {
                'classifier': [MultinomialNB()],
                'classifier__alpha': np.linspace(0.1, 1.0, 10)
            }
        ]

        if mlflow.active_run() is not None:
            mlflow.end_run() 

        with mlflow.start_run(run_name="Model Training"):
            mlflow.log_param('experiment_name', 'spam_classification')
            mlflow.log_param('num_training_samples', X_train.shape[0])
            mlflow.log_param('num_test_samples', X_test.shape[0])

            with mlflow.start_run(nested=True, run_name="RandomizedSearchCV"):
                random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=5, scoring='f1', random_state=2024, n_jobs=-1)
                random_search.fit(X_train, y_train)

                # Log metrics for each fold using cv_results_
                cv_results = random_search.cv_results_
                for i in range(random_search.cv):
                    mlflow.log_metric(f"f1_score_fold_{i}", cv_results[f'split{i}_test_score'].mean())

                best_params_random = random_search.best_params_
                for key, value in best_params_random.items():
                    mlflow.log_param(f'random_search_{key}', value)
                mlflow.log_metric('random_search_best_f1', random_search.best_score_)

            param_grid = {
                    'classifier__n_estimators': [
                        best_params_random.get('classifier__n_estimators', 100) - 50, 
                        best_params_random.get('classifier__n_estimators', 100), 
                        best_params_random.get('classifier__n_estimators', 100) + 50
                    ],
                    'classifier__max_depth': [
                        best_params_random.get('classifier__max_depth', None),
                        best_params_random.get('classifier__max_depth', 10) + 5 if best_params_random.get('classifier__max_depth') is not None else None
                    ],
                    'classifier__min_samples_split': [
                        best_params_random.get('classifier__min_samples_split', 2), 
                        best_params_random.get('classifier__min_samples_split', 2) + 1
                    ],
                    'classifier__min_samples_leaf': [
                        best_params_random.get('classifier__min_samples_leaf', 1), 
                        best_params_random.get('classifier__min_samples_leaf', 1) + 1
                    ]
                }

            # Perform GridSearchCV for fine-tuning
            with mlflow.start_run(nested=True, run_name="GridSearchCV"):
                grid_search = GridSearchCV(random_search.best_estimator_, param_grid, cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Log metrics for each fold in GridSearchCV using cv_results_
                cv_results = grid_search.cv_results_
                for i in range(grid_search.cv):
                    mlflow.log_metric(f"grid_f1_score_fold_{i}", cv_results[f'split{i}_test_score'].mean())

                best_params_grid = grid_search.best_params_
                for key, value in best_params_grid.items():
                    mlflow.log_param(f'grid_search_{key}', value)
                mlflow.log_metric('grid_search_best_f1', grid_search.best_score_)

            # Save the best model after GridSearchCV
            self.model = grid_search.best_estimator_
            
            # Save the train-test split data for evaluation
            self.X_test, self.y_test = X_test, y_test

            # Infer signature for the model
            signature = infer_signature(X_train, self.model.predict(X_train))
            input_example = pd.DataFrame([X_train.iloc[0]], columns=["message"])

            # End any active run before starting a new one for model logging
            if mlflow.active_run():
                mlflow.end_run()
                
            # Log the final model with a new run specifically for model logging
            with mlflow.start_run(run_name="Model Evaluation Logging"):
                # Log the final model with the signature and input example
                mlflow.sklearn.log_model(
                    self.model, 
                    "model", 
                    signature=signature, 
                    input_example=input_example
                )
                
                print("Model artifact logged with MLflow")

                # Evaluate the model on the test set
                y_test_pred = self.model.predict(X_test)
                f1_test = f1_score(y_test, y_test_pred)
                accuracy_test = accuracy_score(y_test, y_test_pred)
                precision_test = precision_score(y_test, y_test_pred)
                recall_test = recall_score(y_test, y_test_pred)

                # Log the evaluation metrics for the test set
                mlflow.log_metric("f1_test", f1_test)
                mlflow.log_metric("accuracy_test", accuracy_test)
                mlflow.log_metric("precision_test", precision_test)
                mlflow.log_metric("recall_test", recall_test)

                # Generate and log confusion matrix for the test set
                cm = confusion_matrix(y_test, y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig("confusion_matrix_test.png")
                mlflow.log_artifact("confusion_matrix_test.png")
            
    def save_model(self, filepath: str ='models/spam_classifier_pipeline.pkl') -> None:
        """
        Saves the trained pipeline (features + classifier) to a file.
        
        Parameters:
            filepath (str): Path to save the model.
        
        Returns:
            None
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved at {filepath}")
