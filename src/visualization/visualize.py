import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from yellowbrick.classifier import (
    ClassificationReport,
    ConfusionMatrix,
    ROCAUC,
    PrecisionRecallCurve,
    ClassPredictionError,
)
from yellowbrick.model_selection import LearningCurve
import pandas as pd
import numpy as np
import mlflow


class Visualizer:

    def plot_message_distribution(self, df: pd.DataFrame) -> None:
        """
        Plots the distribution of spam and ham messages in the dataset and logs the plot as an artifact in MLflow.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'label' column with spam/ham labels.

        Returns:
            None
        """
        # Plot class distribution
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x="label", data=df, palette="viridis")
        plt.title("Message Distribution: Ham vs Spam")
        plt.xlabel("Label")
        plt.ylabel("Count")

        # Add count labels
        for container in ax.containers:
            ax.bar_label(container)

        # Save the plot
        plt.savefig("message_distribution.png")
        plt.show()

        # Log the plot to MLflow as an artifact
        mlflow.log_artifact("message_distribution.png")

    def plot_wordcloud(self, df: pd.DataFrame, label: int, title: str) -> None:
        """
        Generates and displays a word cloud for spam or ham messages and logs the word cloud as an artifact in MLflow.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'cleaned_message' column with preprocessed text data.
            label (int): Label indicating spam (1) or ham (0).
            title (str): Title for the word cloud plot.

        Returns:
            None
        """

        # Separate the spam and ham messages
        spam_messages = df[df["label"] == "spam"]["message"]
        ham_messages = df[df["label"] == "ham"]["message"]

        # Generate word clouds for spam and ham messages
        spam_wordcloud = WordCloud(
            width=600, height=400, background_color="white"
        ).generate(" ".join(spam_messages))
        ham_wordcloud = WordCloud(
            width=600, height=400, background_color="white"
        ).generate(" ".join(ham_messages))

        plt.figure(figsize=(12, 6))

        # Spam word cloud
        plt.subplot(1, 2, 1)
        plt.imshow(spam_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Spam Messages Word Cloud")

        # Ham word cloud
        plt.subplot(1, 2, 2)
        plt.imshow(ham_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Ham Messages Word Cloud")

        # Save the word cloud
        plt.savefig("word_cloud.png")
        plt.show()

        # Log the word cloud to MLflow as an artifact
        mlflow.log_artifact("word_cloud.png")

    def evaluation_report(
        self,
        model: object,
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        label: list = ["ham", "spam"],
    ) -> None:
        """
        Function to plot various visualizations for a classification model in a 3x3 matrix layout, and log the plots as artifacts in MLflow.

        Parameters:
            model (object): The trained classification model.
            X_train (np.array): Training feature set.
            y_train (np.array): Training labels.
            X_test (np.array): Test feature set.
            y_test (np.array): Test labels.
            label (list): List of target class names (default: ['ham', 'spam']).

        Returns:
            None: Plots the visualizations in a 3x3 matrix layout.
        """

        # Create a 2x3 grid for subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.tight_layout(pad=4.0)

        # Create visualizations
        visualizers = [
            ClassificationReport(model, classes=label, ax=axes[0, 0]),
            ConfusionMatrix(model, classes=label, ax=axes[0, 1]),
            ROCAUC(model, classes=label, ax=axes[0, 2]),
            PrecisionRecallCurve(
                model, classes=label, per_class=True, cmap="Set1", ax=axes[1, 0]
            ),
            ClassPredictionError(model, classes=label, ax=axes[1, 1]),
            LearningCurve(model, scoring="f1_weighted", ax=axes[1, 2]),
        ]

        # Fit the model and score for each visualizer
        for viz in visualizers:
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.finalize()

        # Save the evaluation report plot
        plt.savefig("evaluation_report.png")
        plt.show()

        # Log the evaluation report plot to MLflow
        mlflow.log_artifact("evaluation_report.png")
