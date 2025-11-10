# app.py
# Author: Diksha Gole
# Project: Twitter Sentiment Analysis (Multi-class Streamlit App)
# Description: Streamlit dashboard for real-time and dataset-level sentiment prediction (no pie chart version).

import streamlit as st
import joblib
import os
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# NLTK Setup
# ===============================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# ===============================
# Text Cleaning Function
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(filtered)

# ===============================
# Load Model and Vectorizer
# ===============================
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="wide")

st.title("💬 Twitter Sentiment Analysis Dashboard")
st.write("Analyze the sentiment of tweets using Machine Learning — supports Positive, Negative, Neutral, and Irrelevant tweets.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.warning("⚠️ Model files not found. Please train the model first using `train_sentiment_model.py`.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ===============================
# Sidebar Information
# ===============================
st.sidebar.header("📁 About the Project")
st.sidebar.info("""
**Author:** Diksha Gole  
**Project:** Twitter Sentiment Analysis (Multi-Class)  
**Model:** Logistic Regression + TF-IDF  
**Dataset:** Kaggle (twitter_training.csv & twitter_validation.csv)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Sentiment Categories")
st.sidebar.write("✅ **Positive** — happy or favorable tweets")
st.sidebar.write("😐 **Neutral** — balanced or uncertain tweets")
st.sidebar.write("😞 **Negative** — sad or angry tweets")
st.sidebar.write("❔ **Irrelevant** — off-topic or unclear tweets")

# ===============================
# Input Section for Single Tweet
# ===============================
st.markdown("### ✍️ Enter a Tweet for Sentiment Prediction")
user_input = st.text_area("Type or paste your tweet here:", height=120, placeholder="e.g. I absolutely love this new update!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment_map = {
            2: ("Positive 😊", "green"),
            1: ("Neutral 😐", "gray"),
            0: ("Negative 😞", "red"),
            -1: ("Irrelevant ❔", "orange")
        }

        sentiment_label, color = sentiment_map.get(prediction, ("Unknown", "black"))
        st.markdown(f"### 🎯 **Prediction:** <span style='color:{color}'>{sentiment_label}</span>", unsafe_allow_html=True)

# ===============================
# Sample Tweet Analysis
# ===============================
st.markdown("---")
st.subheader("📊 Example Predictions")

sample_tweets = [
    "I absolutely love this new feature!",
    "This is the worst experience ever.",
    "Not sure how I feel about this update.",
    "This post makes no sense at all."
]

if st.button("Run Sample Predictions"):
    results = []
    for text in sample_tweets:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment = {2: "Positive", 1: "Neutral", 0: "Negative", -1: "Irrelevant"}[pred]
        results.append({"Tweet": text, "Sentiment": sentiment})

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

# ===============================
# Analyze Dataset Tweets (Upload)
# ===============================
st.markdown("---")
st.subheader("📂 Test Tweets Directly from Your Dataset")

uploaded_file = st.file_uploader("Upload a CSV file with tweets (optional):", type=["csv"])

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.write("✅ File Loaded Successfully! Columns found:", list(df_test.columns))

        # Auto-detect tweet column
        tweet_col = None
        for col in df_test.columns:
            if 'tweet' in col.lower() or 'text' in col.lower():
                tweet_col = col
                break

        if tweet_col:
            st.success(f"✅ Using '{tweet_col}' as the tweet column.")
            if st.button("Predict Sentiments for Uploaded Dataset"):
                df_test['Cleaned_Tweet'] = df_test[tweet_col].astype(str).apply(clean_text)
                X_test = vectorizer.transform(df_test['Cleaned_Tweet'])
                preds = model.predict(X_test)
                sentiment_labels = {2: "Positive", 1: "Neutral", 0: "Negative", -1: "Irrelevant"}
                df_test['Predicted_Sentiment'] = [sentiment_labels[p] for p in preds]

                st.dataframe(df_test[[tweet_col, 'Predicted_Sentiment']].head(15))

                st.info("✅ Predictions complete. Showing first 15 results.")
        else:
            st.error("❌ Could not find a tweet/text column automatically. Please make sure your CSV has a 'tweet' or 'text' column.")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built by **Diksha Gole** | BSc Data Science Third Year | BSc23DS46 | Minor Project 2025 Semester V")
