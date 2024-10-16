import pytest
from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)


# Test the /predict endpoint for a spam message
def test_predict_spam():
    response = client.post(
        "/predict",
        json={
            "message": "WINNER! Claim your free vacation now and free iPhone now! Rush now!!. Click the link to access your prize."
        },
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"


# Test the /predict endpoint for a ham message
def test_predict_ham():
    response = client.post(
        "/predict", json={"message": "Hey, are you free tomorrow for a meeting?"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "ham"


# Test with an empty message
def test_empty_message():
    response = client.post("/predict", json={"message": ""})
    assert (
        response.status_code == 422
    ), "Expected 422 Unprocessable Entity for empty message"


# Test with non-text input (edge case)
def test_non_text_input():
    response = client.post("/predict", json={"message": 12345})
    assert response.status_code == 422


# Test with a message that could be either spam or ham (ambiguous case)
def test_ambiguous_message():
    response = client.post(
        "/predict", json={"message": "Free lunch tomorrow, join us?"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in ["spam", "ham"]
