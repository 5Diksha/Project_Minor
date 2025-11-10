# train_sentiment_model.py
# Author: Diksha Gole
# Project: Twitter Sentiment Analysis (Multi-class, Final Version)
# Description: Trains and evaluates a sentiment analysis model using Kaggle's Twitter datasets.

import os
import re
import nltk
import pandas as pd
import joblib
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# =======================================
# STEP 1: NLTK SETUP
# =======================================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# =======================================
# STEP 2: LOAD DATASETS
# =======================================
TRAIN_PATH = "twitter_training.csv"
VALID_PATH = "twitter_validation.csv"

if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH):
    print("❌ Dataset files not found! Please make sure 'twitter_training.csv' and 'twitter_validation.csv' are in the same folder.")
    exit()

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

print("✅ Datasets Loaded Successfully!")
print("Training Data Shape:", train_df.shape)
print("Validation Data Shape:", valid_df.shape)

# =======================================
# STEP 3: CLEAN AND PREPROCESS DATA
# =======================================
train_df.columns = ['id', 'topic', 'sentiment', 'tweet']
valid_df.columns = ['id', 'topic', 'sentiment', 'tweet']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(filtered)

train_df['Cleaned_Tweet'] = train_df['tweet'].apply(clean_text)
valid_df['Cleaned_Tweet'] = valid_df['tweet'].apply(clean_text)

# =======================================
# STEP 4: ENCODE SENTIMENT LABELS
# =======================================
sentiment_map = {
    "Positive": 2,
    "Negative": 0,
    "Neutral": 1,
    "Irrelevant": -1
}

train_df = train_df[train_df['sentiment'].isin(sentiment_map.keys())]
valid_df = valid_df[valid_df['sentiment'].isin(sentiment_map.keys())]

train_df['Sentiment'] = train_df['sentiment'].map(sentiment_map)
valid_df['Sentiment'] = valid_df['sentiment'].map(sentiment_map)

# Drop any missing or invalid rows
train_df.dropna(subset=['Cleaned_Tweet', 'Sentiment'], inplace=True)
valid_df.dropna(subset=['Cleaned_Tweet', 'Sentiment'], inplace=True)

print(f"✅ Training samples after cleaning: {len(train_df)}")
print(f"✅ Validation samples after cleaning: {len(valid_df)}")

# =======================================
# STEP 5: FEATURE EXTRACTION
# =======================================
vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['Cleaned_Tweet'])
y_train = train_df['Sentiment']

X_valid = vectorizer.transform(valid_df['Cleaned_Tweet'])
y_valid = valid_df['Sentiment']

# =======================================
# STEP 6: TRAIN MODEL
# =======================================
model = LogisticRegression(max_iter=3000, solver='saga', C=2.0)
model.fit(X_train, y_train)

# =======================================
# STEP 7: EVALUATE MODEL
# =======================================
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print("\n🎯 Model Accuracy on Validation Set:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_valid, y_pred, target_names=["Negative", "Neutral", "Positive", "Irrelevant"], zero_division=0))

# =======================================
# STEP 8: SAVE MODEL & VECTORIZER
# =======================================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and Vectorizer saved successfully!")
print("📁 Files created: model.pkl, vectorizer.pkl")

# =======================================
# STEP 9: QUICK SANITY CHECK
# =======================================
print("\n🔍 Quick Sanity Check:")
sample_tweets = [
    "I absolutely love this new feature!",
    "This is the worst experience ever.",
    "Not sure how I feel about this update.",
    "This post makes no sense at all.",
]

for text in sample_tweets:
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    
    sentiment_label = {
        2: "Positive 😊",
        1: "Neutral 😐",
        0: "Negative 😞",
        -1: "Irrelevant ❔"
    }[pred]
    
    print(f"Tweet: {text} → {sentiment_label}")

