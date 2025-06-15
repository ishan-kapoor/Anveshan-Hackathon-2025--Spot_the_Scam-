# File: dashboard/app.py

import os
import sys
import pandas as pd
import streamlit as st

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_on_dataframe
from dashboard.components.charts import plot_histogram, plot_pie_chart, get_top_suspicious

# ✅ MUST BE FIRST
st.set_page_config(page_title="Spot the Scam", layout="wide")

# 🧠 Page Title
st.title("🔍 Spot the Scam - Job Fraud Detector")

# 💡 Intro Section
st.markdown("""
Upload a CSV with job listings to:
- Predict fraudulent postings
- Visualize fraud probabilities
- Explore suspicious jobs
""")

# 📂 File Upload Section
uploaded_file = st.file_uploader("📁 Upload CSV file with job listings", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Check for required columns
        if not {'title', 'description'}.issubset(input_df.columns):
            st.error("🚫 The uploaded CSV must contain at least 'title' and 'description' columns.")
        else:
            pred_df = predict_on_dataframe(input_df)

            # 📄 Results Preview
            st.subheader("📄 Prediction Results (Top 20 Rows)")
            st.dataframe(pred_df[['text', 'fraud_probability', 'fraud_pred']].head(20))

            # 📊 Histogram
            st.subheader("📊 Fraud Probability Distribution")
            st.pyplot(plot_histogram(pred_df))

            # 🧮 Pie Chart
            st.subheader("🧮 Prediction Class Breakdown")
            st.pyplot(plot_pie_chart(pred_df))

            # 🚩 Top Suspicious Jobs
            st.subheader("🚩 Top 10 Most Suspicious Listings")
            st.write(get_top_suspicious(pred_df))

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
