import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


class ModelEvaluator:

    def evaluate(self, model: BaseEstimator, X_test, y_test) -> None:
        """
        Evaluates the model on the test data, logs metrics and artifacts using MLflow.

        Parameters:
            None

        Returns:
            None
        """
        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Print the classification report
        print(f"Accuracy: {accuracy}")
        print(
             f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}"
         )

        # End any active run before starting a new one for model logging
        if mlflow.active_run():
             mlflow.end_run()

        # # MLflow logging
        with mlflow.start_run(run_name="Evaluate Model"):
             # Log evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

        #     # Optionally: Log the classification report as a text artifact
            classification_rep = classification_report(
                 y_test, y_pred, target_names=["ham", "spam"]
             )
        
            with open("classification_report.txt", "w") as f:
                 f.write(classification_rep)
            mlflow.log_artifact("classification_report.txt")

             # Generate and log confusion matrix for the test set
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("confusion_matrix_test.png")
            mlflow.log_artifact("confusion_matrix_test.png")

            print(f"Model evaluation complete. Metrics logged to MLflow.")
