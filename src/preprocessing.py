# File: src/preprocessing.py

"""
Preprocessing module for job listing text cleaning and transformation.
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Cleans text by removing HTML, special characters, lowercasing.
    """
    if pd.isnull(text):
        return ""
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # Keep letters and spaces
    return text.lower().strip()


def preprocess_dataframe(df):
    """
    Combines 'title' and 'description' into 'text', then cleans it.
    """
    df = df.copy()
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['text'] = (df['title'] + ' ' + df['description']).apply(clean_text)
    return df


def get_vectorizer(train_texts, max_features=5000):
    """
    Fits a TF-IDF vectorizer on input texts.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    return vectorizer


def save_processed_data(raw_path="data/raw/train.csv"):
    """
    Reads raw job data, cleans it, splits it, and saves processed CSVs.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"🚫 Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df = preprocess_dataframe(df)

    if 'fraudulent' not in df.columns:
        raise ValueError("⚠️ 'fraudulent' column not found in the raw CSV.")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraudulent'])

    # Save processed files
    os.makedirs("data/processed", exist_ok=True)
    train_df[['text', 'fraudulent']].to_csv("data/processed/processed_train.csv", index=False)
    test_df[['text', 'fraudulent']].to_csv("data/processed/processed_test.csv", index=False)

    print("✅ Processed data saved to data/processed/")
