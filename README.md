
<h5>Spam Email Classification</h5>

This project is a comprehensive end-to-end solution for classifying SMS messages as either `spam` or `ham` using machine learning. It is designed with the goal of helping users detect unsolicited and potentially harmful messages in real-time using a FastAPI-based web application.

<h5> Table of Contents</h5>

- [Project Details](#project-details)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Deployment](#deployment)
  - [API](#api)
    - [How to Run the API](#how-to-run-the-api)
    - [Request Template](#request-template)
    - [Response](#response)
  - [Streamlit Deployment](#streamlit-deployment)
    - [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
- [Testing](#testing)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

#### Project Details<a name="pd"></a>

<details>
<summary>Problem Statement</summary>
<p>
The task of distinguishing between legitimate messages (ham) and spam is critical to maintaining the integrity of mobile communication. Traditional rule-based systems often fail to effectively adapt to the evolving nature of spam, making machine learning a powerful tool in this context. A well-trained machine learning model can learn the patterns and features that distinguish spam from ham, improving both accuracy and adaptability.
</p>
<p>
In this project, we aim to build a spam classification system that can effectively classify SMS messages as either spam or ham. The model will be trained on the SMS Spam Collection dataset from the UCI Machine Learning Repository, which contains raw SMS text data. Our goal is to leverage text processing and machine learning techniques to develop a robust model that can accurately classify unseen SMS messages.
</p>
</details>


<details>
<summary>Goals</summary>

1. Build a robust pipeline to preprocess raw SMS text messages.
2. Train various machine learning models to classify messages as `spam` or `ham`.
3. Deploy a real-time classification model using a FastAPI web app, allowing users to interact with the model.

</details>

<details>
<summary>Methodology</summary>

The project follows a pipeline-based approach:

1. **`Data Collection`**: The dataset used for this project is the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from the UCI Machine Learning Repository.
2. **`Exploratory Data Analysis (EDA)`**: We perform EDA to understand the dataset and extract useful insights, including message distribution, word frequency, etc.
3. **`Text Processing`**: Text normalization, tokenization, stopword removal, stemming, and lemmatization are used to preprocess the data.
4. **`Feature Extraction`**: Using TF-IDF to convert the text data into numerical features for the models.
5. **`Modeling`**: Training models such as Naive Bayes, Logistic Regression, and Random Forest.
6. **`Model Evaluation`**: Evaluation using metrics like accuracy, F1 score, and ROC-AUC.
7. **`Deployment`**: The model is deployed using FastAPI and can be run as a Docker container.

</details>

<details>
<summary>Exploratory Data Analysis (EDA)</summary>

Key insights from the EDA:
- The dataset contains `5,572` messages, of which `13.4%` are labeled as `spam`.
- `Spam messages` tend to be longer on average than `ham messages`.
- The most frequent words in spam messages were `free`, `won`, `urgent`, `prize` and `claim` stand out, which are typical indicators of spam.

</details>

<details>
<summary>Text Processing</summary>

1. **`Normalization`**: Lowercasing, removing punctuation, and special characters.
2. **`Tokenization`**: Breaking the messages into words.
3. **`Stopword Removal`**: Removing common words that don’t contribute much meaning such as "the", "is", etc.
4. **`Lemmatization`**: Reducing words to their root form for consistent analysis.

</details>

<details>
<summary>Feature Extraction</summary>

We used `TF-IDF (Term Frequency-Inverse Document Frequency)` to convert the preprocessed text into numerical feature vectors that are fed into the machine learning models.

</details>

<details>
<summary>Modeling</summary>

The following machine learning models below were trained and evaluated. Also, we used `RandomizedSearchCV` and `GridSearchCV` to fine-tune model hyperparameters.

- Naive Bayes
- Logistic Regression
- Random Forest

The best-performing model was `Random Forest` with an F1 score of `0.93`. Below is report of the performance of the model.

<img src="/model_evaluation_report.png" />

<h5>Summary</h5>

The spam classification model was developed to distinguish between `ham` and `spam` messages, an essential task in filtering unsolicited content and ensuring smooth email communications. The model performs exceptionally well in classifying `ham` messages, with high `precision 0.976` and perfect `recall 1.0`, ensuring that nearly all legitimate messages are correctly identified.

For Spam messages, the model achieves a perfect `precision 1.0`, meaning it rarely misclassifies legitimate messages as `spam (false positives)`. However, the `recall 0.839` indicates that about `18%` of actual spam messages are missed, resulting in false negatives. The overall `F1-score 0.912` for `spam` reflects a good balance between precision and recall.

<br>
<h5>Real-World Implications</h5>

In practical applications, this approach is beneficial, as the cost of misclassifying a legitimate email as `spam (false positive)` is higher than missing a few spam messages `(false negative)`. Users would rather receive an occasional spam message in their inbox than risk missing important communication. The model's focus on minimizing false positives aligns with this priority.

</details>

#### Setup Instructions<a name="si"></a>

##### Prerequisites<a name="pr"></a>

- Python 3.8+
- Git
- Docker (optional for containerized deployment)

##### Installation<a name="is"></a>

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/spam_email_classification.git
   cd spam_email_classification
   ```

2. Create a virtual environment

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. Install the dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Run the app with Docker

   ```bash
   docker build -t spam_classifier_app .
   docker run -p 8000:8000 spam_classifier_app
   ```

#### Deployment<a name="dt"></a>

You can deploy the spam classification model using two different modes: `API` and `Streamlit`.

##### API
The API is built using `FastAPI` and allows you to send requests for spam classification predictions.

###### How to Run the API
1. Train and save model. You can view model parameters and metrics on `mlflow ui` and pipeline on `zenml ui`
   ```bash
   python run_pipeline.py
   ```
2. Ensure the model is saved in `models` folder and the API script `api.py` is available in `app` folder.
3. Run the API script. The API will start at http://0.0.0.0:8000
   ```bash
   python app/api.py
   ```
4. Once the api is running, you can interact with the model by sending POST requests to `/predict` using template below.

###### Request Template<a name="rt"></a>

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Congratulations! You have won a free ticket to Bahamas. Claim now!"
  }'

```

###### Response<a name="rs"></a>

```json
{
  "prediction": "spam"
}
```
<br>

##### Streamlit Deployment
The Streamlit app allows you to interact with the model through a web interface and input text messages for spam classification.

###### How to Run the Streamlit App
1. Train and save model. You can view model parameters and metrics on `mlflow ui` and pipeline on `zenml ui`
   ```bash
   python run_pipeline.py
   ```
2. Ensure the model is saved in `models` folder and the app script `spam_app.py` is available in `app` folder.
3. Run the app script. The app will be accessible in your browser at http://localhost:8501
   ```bash
   python app/spam_app.py
   ```
4. This will open a web interface where you can interact with.
5. Enter a message in the text box provided and click the classify button to get the prediction (Spam or Ham).

#### Testing<a name="tt"></a>

Unit tests are included for data loading, preprocessing, feature extraction, and model prediction. You can run tests using `pytest`:

```bash
pytest tests/
```

#### Technologies Used<a name="tu"></a>

- Python
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn
- MLflow
- Docker
- ZenMl
- Yellowbrick
- NLTK

#### Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.