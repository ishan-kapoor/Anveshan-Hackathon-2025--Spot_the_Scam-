# 🕵️‍♂️ Spot the Scam - Job Fraud Detection

**By Team: Info Inquisitors**

**Members:**
- **Ishan Kapoor** – ishankpor@gmail.com  
- **Malya Kapoor** – malyakapoor4@gmail.com  

---

## 🧠 Project Overview

**Spot the Scam** is an intelligent fraud detection system designed to identify potentially fake job listings. With the rise of online employment scams, our solution aims to analyze job postings using machine learning and natural language processing (NLP) to predict the probability of fraud. 

The system is suitable for:
- Job boards verifying listings
- HR teams screening job ads
- Applicants avoiding scam jobs

---

## 🚀 Key Features & Technologies Used

### ✅ Features
- Clean and preprocess raw job data (title + description)
- Train Logistic Regression or Random Forest models
- Predict fraud probability on new/unseen listings
- Visualize:
  - Distribution of fraud probabilities (histogram)
  - Prediction class breakdown (genuine vs fraud)
  - Top 10 most suspicious job listings
- Retrain model easily on new labeled data

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
```bash
git clone https://github.com/your-username/spot-the-scam.git
cd spot-the-scam

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
