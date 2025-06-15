# File: train_and_save_vectorizer.py

import pandas as pd
import joblib
from src.preprocessing import preprocess_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training data
df = pd.read_csv("data/processed/processed_train.csv")

# Preprocess (optional, if already processed then skip this)
df = preprocess_dataframe(df)

# Fit vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df["text"])

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Vectorizer saved to models/vectorizer.pkl")
