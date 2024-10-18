import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from yellowbrick.classifier import (
    ClassificationReport,
    ConfusionMatrix,
    ROCAUC,
    PrecisionRecallCurve,
    ClassPredictionError,
)
from yellowbrick.model_selection import LearningCurve
import pandas as pd
from typing import List


class ModelEvaluator:

    def evaluate(
        self,
        model: BaseEstimator,
        X_train: pd.Series,
        y_train: pd.Series,
        X_test: pd.Series,
        y_test: pd.Series,
        labels: List[str] = ["ham", "spam"],
    ) -> None:
        """
        Evaluates the trained model on test data, generates visualizations, and logs metrics and artifacts using MLflow.

        Parameters:
            model (BaseEstimator): The trained classification model (pipeline).
            X_train (pd.Series): The training feature set.
            y_train (pd.Series): The training labels.
            X_test (pd.Series): The test feature set.
            y_test (pd.Series): The test labels.
            labels (List[str]): A list of class labels for the classification task, by default ["ham", "spam"].

        Returns:
            None
        """

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Print the classification report
        print(f"Accuracy: {accuracy}")
        print(
            f"Classification Report:\n{classification_report(y_test, y_pred, target_names=labels)}"
        )

        # Log evaluation metrics
        with mlflow.start_run(nested=True, run_name="Evaluate Model"):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Create a 2x3 grid for subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            plt.tight_layout(pad=4.0)

            # Create and log visualizations using Yellowbrick
            visualizers = [
                ClassificationReport(model, classes=labels, ax=axes[0, 0]),
                ConfusionMatrix(model, classes=labels, ax=axes[0, 1]),
                ROCAUC(model, classes=labels, ax=axes[0, 2]),
                PrecisionRecallCurve(
                    model, classes=labels, per_class=False, cmap="Set1", ax=axes[1, 0]
                ),
                ClassPredictionError(model, classes=labels, ax=axes[1, 1]),
                LearningCurve(model, scoring="f1_weighted", ax=axes[1, 2]),
            ]

            # Fit and score each visualizer
            for viz in visualizers:
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.finalize()

            # Save the visualizations as a PNG and log as artifact
            plt.savefig("model_evaluation_report.png")
            mlflow.log_artifact("model_evaluation_report.png")

            print(
                "Model evaluation complete. Metrics and visualizations logged to MLflow."
            )
