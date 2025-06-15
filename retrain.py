# File: retrain.py

"""
Script to retrain model on updated labeled data (train.csv).
Useful when new data is added to data/raw/train.csv.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_dataframe
from src.model import train_model, save_model, save_vectorizer

TRAIN_PATH = "data/raw/train.csv"
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def retrain(model_type='logistic'):
    print("🔄 Loading training data...")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"🚫 File not found: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)

    print("🧼 Preprocessing data...")
    processed_df = preprocess_dataframe(df)
    X_text = processed_df["text"]
    y = processed_df["fraudulent"]

    print("🧠 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X_text)

    print(f"🤖 Training {model_type} model...")
    model = train_model(X_vectorized, y, model_type=model_type)

    print("💾 Saving model and vectorizer...")
    save_model(model, MODEL_PATH)
    save_vectorizer(vectorizer, VECTORIZER_PATH)

    print("✅ Retraining complete! Model and vectorizer saved.")


if __name__ == "__main__":
    retrain()
