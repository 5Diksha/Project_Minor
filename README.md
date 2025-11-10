# Project_Minor
X (twitter) Sentimental Analysis App
# Twitter Sentiment Analysis

**Author:** Diksha Gole  
**Description:** End-to-end sentiment analysis project for tweets. Preprocesses tweets, trains a TF-IDF + Logistic Regression model, and provides a Streamlit app for live predictions.

## Files
- `train_sentiment_model.py` : Downloads dataset (if needed), trains model, saves `model.pkl` and `vectorizer.pkl`.
- `app.py` : Streamlit app that loads the saved model and predicts sentiment for user input.
- `twitter_sentiment.csv` : Dataset (auto-downloaded by training script if missing).
- `model.pkl`, `vectorizer.pkl` : Generated after running training script.

## Setup & Run (local / VS Code)
1. Open project folder in VS Code.
2. (Optional) create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
