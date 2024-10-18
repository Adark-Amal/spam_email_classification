from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import logging
import uvicorn


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load("models/spam_classifier_pipeline.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


class MessageInput(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Message cannot be empty or too long.",
    )


# Define the prediction endpoint
@app.post("/predict")
def predict_spam(input: MessageInput):
    """
    Predicts whether the given message is spam or ham.

    Parameters:
        input (MessageInput): Input message sent in the request body.

    Returns:
        dict: Prediction result as either 'spam' or 'ham'.
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Try again later."
        )

    # Extract the message from the input
    message = input.message

    try:
        # Use the model to predict
        prediction = model.predict([message])

        # Return the result as a JSON response
        result = "spam" if prediction[0] == 1 else "ham"
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try again later."
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)