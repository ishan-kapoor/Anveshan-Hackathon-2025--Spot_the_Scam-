# File: src/utils.py

"""
Utility functions for saving, loading, and visualizing prediction results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_csv(df, path):
    """
    Save a DataFrame to CSV at the specified path.
    Creates the folder if it doesn't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ CSV saved to {path}")


def load_csv(path):
    """
    Load a CSV file into a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚫 File not found: {path}")
    return pd.read_csv(path)


def plot_histogram(data, path="outputs/figures/histogram.png"):
    """
    Plot and save histogram of fraud probability distribution.
    """
    if "fraud_probability" not in data.columns:
        raise ValueError("Missing 'fraud_probability' column in DataFrame.")

    plt.figure(figsize=(8, 6))
    sns.histplot(data['fraud_probability'], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of Fraud Probabilities")
    plt.xlabel("Probability of Fraud")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"📊 Histogram saved to {path}")


def plot_pie_chart(data, path="outputs/figures/pie_chart.png"):
    """
    Plot and save pie chart of predicted class distribution.
    """
    if "fraud_pred" not in data.columns:
        raise ValueError("Missing 'fraud_pred' column in DataFrame.")

    plt.figure(figsize=(6, 6))
    labels = ['Genuine', 'Fraudulent']
    counts = data['fraud_pred'].value_counts().sort_index()

    if len(counts) < 2:  # One class only
        counts = counts.reindex([0, 1], fill_value=0)

    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    plt.title("Prediction Class Distribution")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"🥧 Pie chart saved to {path}")
