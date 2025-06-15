# File: src/model.py

"""
Model training, evaluation, saving and loading utilities.
"""

import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def train_model(X_train, y_train, model_type='logistic'):
    """
    Train a classification model.
    
    Parameters:
    - X_train, y_train: training data
    - model_type: 'logistic' (default) or 'random_forest'
    
    Returns:
    - trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced', max_iter=500)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
    else:
        raise ValueError("❌ Unsupported model type. Choose 'logistic' or 'random_forest'.")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate a trained model using F1-score.
    
    Returns:
    - f1 score (float)
    """
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)


def save_model(model, path="models/best_model.pkl"):
    """
    Save trained model to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {path}")


def load_model(path="models/best_model.pkl"):
    """
    Load a pickled model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚫 Model not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_vectorizer(vectorizer, path="models/vectorizer.pkl"):
    """
    Save the TF-IDF vectorizer to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved to {path}")


def load_vectorizer(path="models/vectorizer.pkl"):
    """
    Load the saved TF-IDF vectorizer from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚫 Vectorizer not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
