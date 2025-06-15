# 🕵️‍♂️ Spot the Scam - Job Fraud Detection

Welcome to the **Spot the Scam** project, A full stack data science solution that detects **fraudulent job postings** using Natural Language Processing (NLP) and machine learning.

**By Team: Info Inquisitors**

**Members:**
- **Ishan Kapoor** – ishankpor@gmail.com  ( iitg_aiml_2503897 )
- **Malya Kapoor** – malyakapoor4@gmail.com  ( iitg_aiml_25030028 )

👉 [GitHub Repository](https://github.com/ishan-kapoor/Anveshan-Hackathon-2025--Spot_the_Scam-) 
👉 [Project Live URL](https://team-info-inquisitors-spot-the-scam.streamlit.app/) 
👉 [Project Video Presentation Link](https://drive.google.com/file/d/12O5V36aN6724aygcHvi0c_NHYfytJCe_/view?usp=sharing) 

---

## 🧠 Project Overview

**Spot the Scam** is an intelligent fraud detection system designed to identify potentially fake job listings. It provides an **interactive Streamlit dashboard** where users can upload CSVs of job listings and receive predictions on whether each listing is likely to be **fraudulent or legitimate**.

It combines:
- NLP preprocessing
- ML model inference
- Data visualization
- User-friendly UI

---

### 🚀 Key Features & Technologies Used

#### ✅ Features
- Upload CSVs of job listings with `title` and `description`
- Predict and visualize fraud probabilities
- Flag top suspicious listings
- Visual analytics: histogram, pie chart, data table

#### ⚙️ Tech Stack
- Python
- Streamlit (Dashboard)
- scikit-learn (ML model)
- Pandas, NumPy (Data handling)
- Matplotlib / Seaborn (Plots)
- Pickle (Model persistence)

### 🛠️ Tech Stack
- **Python 3.10+**
- **Scikit-learn** – ML model training
- **Pandas** – Data manipulation
- **TfidfVectorizer** – Text feature engineering
- **Matplotlib & Seaborn** – Visualizations
- **Streamlit** – Interactive dashboard
- **Joblib/Pickle** – Model persistence

---

## ⚙️ Setup Instructions

Follow these steps to get the project running on your local machine:

### 1. **Clone the Repository**
- git clone https://github.com/ishan-kapoor/Anveshan-Hackathon-2025--Spot_the_Scam-
- cd spot-the-scam

### 2. **Install Required Packages**
pip install -r requirements.txt

### 3. **Preprocess the Raw Data**
python src/preprocessing.py
This reads data/train.csv and creates:
 - data/processed/processed_train.csv
 - data/processed/processed_test.csv

### 4. **Train the Model**
python retrain.py

### 5. **Run Predictions on Test Set**
python main.py

### 6. **Launch the Streamlit Dashboard**
streamlit run dashboard/app.py

## 📁 Project Structure
spot-the-scam/
│
├── data/
│   ├── raw/train.csv
│   └── processed/
│       ├── processed_train.csv
│       └── processed_test.csv
│
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── outputs/
│   ├── predictions.csv
│   └── figures/
│       ├── histogram.png
│       └── pie_chart.png
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── predict.py
│   └── utils.py
│
├── dashboard/
│   ├── app.py
│   └── components/
│       └── charts.py
│
├── retrain.py
├── main.py
├── requirements.txt
└── README.md

## 🧠 Project Video Presentation
- Link: https://drive.google.com/file/d/12O5V36aN6724aygcHvi0c_NHYfytJCe_/view?usp=sharing
