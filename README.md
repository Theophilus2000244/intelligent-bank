# intelligent-bank


Here is a **professional README file** for your **Fraud Detection Project using XGBoost deployed with a REST API** based on the code you uploaded. You can use this for **GitHub, portfolio, or interview presentations**.

---

# Fraud Detection System using XGBoost with REST API Deployment

## Project Overview

This project develops a **machine learning fraud detection system** that predicts whether a financial transaction is fraudulent or legitimate. The model is trained using **XGBoost**, combined with a custom **feature engineering and feature selection pipeline**, and deployed using a **REST API built with Flask**.

The system allows users or applications to send transaction data to an API endpoint and receive fraud predictions in real time.

---

# Project Architecture

```
Transaction Data
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Feature Engineering Pipeline
       │
       ▼
Feature Selection
       │
       ▼
XGBoost Model Training
       │
       ▼
Model Serialization (Joblib)
       │
       ▼
Flask REST API Deployment
       │
       ▼
Real-time Fraud Prediction
```

---

# Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Model Serialization

* Joblib

### API Development

* Flask

### Data Storage

* CSV

### Deployment Architecture

* REST API

---

# Python Libraries Used

```python
flask
pandas
numpy
joblib
statistics
csv
os
```

### Machine Learning Libraries

```python
xgboost
scikit-learn
```

---

# Custom Pipelines

The project includes custom pipeline components for preprocessing.

### Feature Engineering

```
FraudFeatureEngineer
```

Responsible for:

* Creating new fraud-related features
* Transforming raw transaction data
* Preparing model-ready features

---

### Feature Selection

```
ColumnSelector
```

Responsible for:

* Selecting relevant features
* Ensuring model consistency between training and prediction

---

# Machine Learning Model

### Model Used

XGBoost Classifier

### Why XGBoost?

* High performance on tabular data
* Handles class imbalance well
* Strong predictive power
* Works well for fraud detection problems

---

# Model Pipeline

The prediction pipeline follows these steps:

1. Receive transaction data
2. Convert input JSON to DataFrame
3. Apply Feature Engineering
4. Align columns with model features
5. Apply Feature Selection
6. Predict fraud probability
7. Apply decision threshold
8. Return prediction via API

---

# API Endpoints

## Root Endpoint

```
GET /
```

Returns available API endpoints.

---

## View Transactions

```
GET /transactions
```

Returns a list of stored transactions.

Optional parameter:

```
/transactions?limit=10
```

---

## Get Single Transaction

```
GET /transactions/<id>
```

Returns a specific transaction.

---

## Dataset Statistics

```
GET /stats
```

Returns summary statistics of numeric columns.

---

## Fraud Prediction Endpoint

```
POST /predict
```

Accepts transaction data and returns fraud prediction.

### Example Request

```json
{
  "features": {
    "amount": 250,
    "transaction_type": "online",
    "merchant_id": 1234
  }
}
```

### Example Response

```json
{
  "prediction": 1,
  "fraud_probability": 0.87,
  "threshold_used": 0.5
}
```

Where:

* **1 = Fraud**
* **0 = Legitimate**

---

# Model Artifacts

The API loads the following serialized objects:

| File                         | Purpose                      |
| ---------------------------- | ---------------------------- |
| feature_engineering.pkl      | Feature engineering pipeline |
| feature_selection.pkl        | Feature selection pipeline   |
| xgb_model_with_threshold.pkl | Trained XGBoost model        |
| model_feature_names.pkl      | List of expected features    |

---

# Running the API

Start the Flask server:

```bash
python app.py
```

The API runs on:

```
http://localhost:5000
```

---

# Example Prediction using CURL

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": {"amount": 300}}'
```

---

# Key Skills Demonstrated

* Machine Learning Model Development
* Fraud Detection Modelling
* Feature Engineering
* Feature Selection
* Model Serialization
* REST API Development
* Model Deployment
* End-to-End Data Science Pipeline

---

# Possible Improvements

* Docker containerization
* Cloud deployment (AWS / Azure)
* Model monitoring
* Real-time streaming predictions
* Model retraining pipeline

---
