import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.base import BaseEstimator


class ModelTrainer:

    def random_search_cv(self, pipeline: BaseEstimator, X_train: pd.Series, y_train: pd.Series) -> Tuple[Dict, BaseEstimator]:
        """
        Perform random search of hyperparameters
        
        Parameters:

        
        Returns:
            Best estimator
        """
        
        # Define parameter space for random and grid search
        param_distributions = [
            {
                "classifier": [RandomForestClassifier(random_state=2024)],
                "classifier__n_estimators": np.arange(50, 200, 50),
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": np.arange(2, 10),
                "classifier__min_samples_leaf": [1, 2, 4],
            },
            {
                "classifier": [LogisticRegression(random_state=2024)],
                "classifier__C": np.logspace(-3, 3, 7),
                "classifier__solver": ["lbfgs", "liblinear"],
            },
            {
                "classifier": [MultinomialNB()],
                "classifier__alpha": np.linspace(0.1, 1.0, 10),
            },
        ]

        # Create sub runs for random
        with mlflow.start_run(nested=True, run_name="RandomizedSearchCV"):
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=15,
                cv=5,
                scoring="accuracy",
                random_state=2024,
                n_jobs=-1,
            )
            
            random_search.fit(X_train, y_train)

            # Log metrics for each fold using cv_results_
            cv_results = random_search.cv_results_
            for i in range(random_search.cv):
                mlflow.log_metric(
                    f"f1_score_fold_{i}", cv_results[f"split{i}_test_score"].mean()
                )

            best_params_random = random_search.best_params_
            best_estimator_random = random_search.best_estimator_

            for key, value in best_params_random.items():
                mlflow.log_param(f"random_search_{key}", value)
            mlflow.log_metric("random_search_best_f1", random_search.best_score_)
        
        return best_params_random, best_estimator_random

    def grid_search_cv(self, random_param: Dict, random_estimator: BaseEstimator, X_train, y_train):
        """
        Perform random search of hyperparameters
        
        Parameters:

        
        Returns:
            model: Pipeline[BaseEstimator, ClassifierMixin]
        """
        
        # Define parameter space for grid search base on results of random search
        param_grid = {
            "classifier__n_estimators": [
                random_param.get("classifier__n_estimators", 100) - 50,
                random_param.get("classifier__n_estimators", 100),
                random_param.get("classifier__n_estimators", 100) + 50,
            ],
            "classifier__max_depth": [
                random_param.get("classifier__max_depth", None),
                (
                    random_param.get("classifier__max_depth", 10) + 5
                    if random_param.get("classifier__max_depth") is not None
                    else None
                ),
            ],
            "classifier__min_samples_split": [
                random_param.get("classifier__min_samples_split", 2),
                random_param.get("classifier__min_samples_split", 2) + 1,
            ],
            "classifier__min_samples_leaf": [
                random_param.get("classifier__min_samples_leaf", 1),
                random_param.get("classifier__min_samples_leaf", 1) + 1,
            ],
        }

        # Perform GridSearchCV for fine-tuning
        with mlflow.start_run(nested=True, run_name="GridSearchCV"):
            grid_search = GridSearchCV(
                random_estimator,
                param_grid,
                cv=10,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            # Log metrics for each fold in GridSearchCV using cv_results_
            cv_results = grid_search.cv_results_
            for i in range(grid_search.cv):
                mlflow.log_metric(
                    f"grid_f1_score_fold_{i}",
                    cv_results[f"split{i}_test_score"].mean(),
                )

            best_params_grid = grid_search.best_params_
            for key, value in best_params_grid.items():
                mlflow.log_param(f"grid_search_{key}", value)
            mlflow.log_metric("grid_search_best_f1", grid_search.best_score_)

        # Save the best model after GridSearchCV
        model = grid_search.best_estimator_
        
        return model
    
    def create_pipeline(self) -> BaseEstimator:
        """
        Create model training pipeline
        
        Parameters:
            None
        
        Returns:
            pipeline
        """
        
        # Define training pipeline
        pipeline = Pipeline(
            [
                ("vectorize", TfidfVectorizer()),
                ("classifier", RandomForestClassifier(random_state=2024)),
            ]
        )
        
        return pipeline
        
    def train(self, X_train: pd.Series, y_train: pd.Series) -> BaseEstimator:
        """
        Trains the model using RandomizedSearchCV followed by GridSearchCV.

        Parameters:
            message_data (pd.DataFrame): DataFrame to be used for training the model.

        Returns:
            best_model: Pipeline[BaseEstimator, ClassifierMixin]
        """
         # End any active run before starting a new one for model logging
        if mlflow.active_run():
            mlflow.end_run()

        # Log the final model with the signature and input example
        with mlflow.start_run(run_name="Train Model"):

            # Define training pipeline
            pipeline = Pipeline(
                [
                    ("vectorize", TfidfVectorizer()),
                    ("classifier", RandomForestClassifier(random_state=2024)),
                ]
            )
            
            random_param, random_model = self.random_search_cv(pipeline, X_train, y_train)
            best_model = self.grid_search_cv(random_param, random_model, X_train, y_train)
            
            # Infer signature for the model
            signature = infer_signature(X_train, best_model.predict(X_train))
            #input_example = pd.DataFrame([X_train.iloc[0]], columns=["message"])

            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                #input_example=input_example,
            )
            
            print("Model artifact logged with MLflow")
        
        return best_model

    def write_model(self, model: BaseEstimator, file_path: str) -> None:
        """
        Saves the trained pipeline (features + classifier) to a file.

        Parameters:
            filepath (str): Path to save the model.

        Returns:
            None
        """
        joblib.dump(model, file_path)
        print(f"Model saved at {file_path}")
