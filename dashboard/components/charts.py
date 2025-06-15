# File: dashboard/components/charts.py

"""
Reusable chart components for the Spot the Scam dashboard.
Includes histogram, pie chart, and suspicious listings table.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Fraud probability histogram
def plot_histogram(pred_df):
    if 'fraud_probability' not in pred_df.columns:
        raise ValueError("❌ Column 'fraud_probability' not found in DataFrame")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(pred_df['fraud_probability'], bins=20, kde=True, color='orange', ax=ax)
    ax.set_title("Fraud Probability Distribution")
    ax.set_xlabel("Probability of Fraud")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

# Pie chart of predicted classes
def plot_pie_chart(pred_df):
    if 'fraud_pred' not in pred_df.columns:
        raise ValueError("❌ Column 'fraud_pred' not found in DataFrame")

    fig, ax = plt.subplots()
    counts = pred_df['fraud_pred'].value_counts().sort_index()

    labels = ['Genuine', 'Fraudulent']
    # If missing class (e.g. only 0s or 1s), adjust labels dynamically
    if len(counts) < 2:
        labels = ['Genuine'] if counts.index[0] == 0 else ['Fraudulent']

    colors = ['#2ecc71', '#e74c3c']
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(counts)], startangle=90)
    ax.set_title("Fraudulent vs Genuine Predictions")
    fig.tight_layout()
    return fig

# Table of top N suspicious listings
def get_top_suspicious(pred_df, top_n=10):
    if 'fraud_probability' not in pred_df.columns or 'text' not in pred_df.columns:
        raise ValueError("❌ Required columns ('text', 'fraud_probability') not found.")
    
    return pred_df.sort_values("fraud_probability", ascending=False).head(top_n)[['text', 'fraud_probability']]
