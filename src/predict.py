# File: src/predict.py

"""
Prediction module for scoring job listings using trained model and vectorizer.
"""

import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.model import load_model, load_vectorizer


def predict_on_dataframe(df):
    """
    Takes a Pandas DataFrame with job listings,
    returns the DataFrame with fraud probabilities and predictions.
    """
    df = preprocess_dataframe(df)

    vectorizer = load_vectorizer("models/vectorizer.pkl")
    model = load_model("models/best_model.pkl")

    X = vectorizer.transform(df["text"])
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_pred"] = model.predict(X)

    return df
