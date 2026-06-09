# 📱 Mobile Sentiment Analysis Dashboard

A Machine Learning web application that predicts customer sentiment toward mobile phones based on review-related features and device information.

Built using **Python**, **Scikit-Learn**, **CatBoost**, and **Streamlit**.

---

## 🚀 Project Overview

Customer reviews contain valuable insights about user satisfaction and product experience. This project analyzes mobile phone review data and predicts whether customer sentiment is:

* 😊 Positive
* 😐 Neutral
* 😡 Negative

The application provides:

* Dataset Overview Dashboard
* Interactive Sentiment Prediction Page
* Machine Learning Model Performance Analysis
* Real-Time Prediction Interface

---

## 📊 Dataset Information

The dataset contains **50,000 customer reviews** collected from multiple e-commerce platforms.

### Features Used

| Feature           | Description               |
| ----------------- | ------------------------- |
| age               | Customer age              |
| brand             | Mobile phone brand        |
| model             | Mobile phone model        |
| price_usd         | Product price in USD      |
| country           | Customer country          |
| language          | Review language           |
| verified_purchase | Verified purchase status  |
| review_length     | Length of review          |
| word_count        | Number of words in review |
| source            | Review platform           |

### Target Variable

| Sentiment |
| --------- |
| Positive  |
| Neutral   |
| Negative  |

---

## 🧹 Data Preprocessing

Several preprocessing steps were applied:

* Removed unnecessary identifiers
* Removed data leakage features
* Feature scaling using RobustScaler
* One-Hot Encoding for categorical variables
* Binary Encoding for high-cardinality features
* Pipeline-based preprocessing workflow

Removed columns:

* review_id
* customer_name
* currency
* review_date
* price_local
* exchange_rate_to_usd
* rating
* battery_life_rating
* camera_rating
* performance_rating
* design_rating
* display_rating
* review_text
* helpful_votes

---

## 🤖 Machine Learning Models

The following models were evaluated using 5-Fold Cross Validation:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* CatBoost Classifier
* LightGBM

### Final Selected Model

**CatBoost Classifier**

Hyperparameter tuning was performed using RandomizedSearchCV.

---

## 📈 Model Performance

### Classification Report

| Class    | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Negative | 0.90      | 0.73   | 0.80     |
| Neutral  | 0.95      | 0.81   | 0.88     |
| Positive | 0.88      | 1.00   | 0.93     |

### Overall Metrics

* Accuracy: 90%
* Weighted F1 Score: 89%
* Macro F1 Score: 87%

---

## 🖥️ Streamlit Application

### Overview Page

Provides:

* Dataset Summary
* Key Performance Indicators
* Dataset Preview
* Sentiment Distribution Visualization

### Prediction Page

Allows users to:

* Select mobile phone information
* Enter review-related characteristics
* Generate real-time sentiment predictions
* View sentiment classification results

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* Category Encoders
* Plotly
* Streamlit
* Joblib

---

## 📂 Project Structure

```bash
Mobile-Sentiment-Analysis/
│
├── app.py
├── pages/
│   ├── 1_Overview.py
│   └── 2_Model_Prediction.py
│
├── MobileReviews_model.pkl
├── Mobile Reviews Sentiment.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Mobile-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 👨‍💻 Developer

**Yasser**

Mobile Sentiment Analysis Dashboard powered by Machine Learning and CatBoost.
