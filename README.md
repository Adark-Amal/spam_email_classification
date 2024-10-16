# SMS Spam Classifier

This project builds an SMS spam classifier using machine learning techniques, including preprocessing, feature extraction (TF-IDF), and classification models (Random Forest). The project also includes a FastAPI-based RESTful API and a Streamlit app for real-time spam classification.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Training](#model-training)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Running the FastAPI Spam Classifier API](#running-the-fastapi-spam-classifier-api)
- [Dockerization](#dockerization)
- [API Usage](#api-usage)
- [Deployment](#deployment)

---

## Project Structure

```bash
├── data/
│   ├── raw/               # Raw dataset
│   ├── processed/          # Processed dataset
├── notebooks/
│   └── spam_filter.ipynb   # Jupyter notebook for EDA and experimentation
├── src/
│   ├── data/
│   │   ├── load_data.py    # Data loading module
│   │   └── preprocess.py   # Preprocessing module (tokenization, stop words, etc.)
│   ├── features/
│   │   └── build_features.py # TF-IDF feature extraction with pipelines
│   ├── models/
│   │   ├── train_model.py  # Model training using pipeline (TF-IDF + classifier)
│   │   └── evaluate_model.py # Model evaluation (accuracy, precision, recall)
│   ├── visualization/
│   │   └── visualise.py    # Visualization scripts
│   ├── utils/
│   │   └── helper_functions.py # Utility functions
├── app/
│   └── streamlit_app.py    # Streamlit app for real-time spam classification
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Dockerfile for deployment
├── README.md               # Project documentation
