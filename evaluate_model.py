# File: evaluate_model.py

import pandas as pd
from sklearn.metrics import f1_score
from src.model import load_model, load_vectorizer
from src.preprocessing import preprocess_dataframe

# 🔹 Load your labeled test data CSV (must include 'title', 'description', 'fraudulent')
df = pd.read_csv("data/raw/train.csv")  # ✅ CHANGE if your test data is elsewhere
y_true = df["fraudulent"]

# 🔹 Preprocess
df = preprocess_dataframe(df)

# 🔹 Load model and vectorizer
model = load_model("models/best_model.pkl")
vectorizer = load_vectorizer("models/vectorizer.pkl")

# 🔹 Transform and Predict
X = vectorizer.transform(df["text"])
y_pred = model.predict(X)

# 🔹 F1 Score
f1 = f1_score(y_true, y_pred)
print(f"✅ F1 Score: {f1:.4f}")
