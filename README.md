# Anti-Fraud Solution ML

This project presents an exploratory data analysis (EDA) and a basic machine learning model for fraud detection in transactional data.

## Highlights

The analysis identifies key fraud patterns such as:

- Strong fraud concentration among a small subset of merchants
- Higher fraud occurrence in high-value transactions
- Temporal fraud patterns, especially during evening hours
- Behavioral signals such as multi-device usage

A Logistic Regression model was developed to predict fraudulent transactions based on these patterns.

## 1) Project Structure

```
project/  
│
├── data/
│   └── transactional-sample.csv
│
├── src/
│   ├── logistic_regression.py
│   └── functions.py
│
├── results/
│   ├── ROC Curve Plot.png
│   └── Model Classification Report and Decision Engine Count.png
│ 
├── notebooks/
│   └── eda.ipynb
│
├── report/
│   └── risk_report.pdf
│ 
├── requirements.txt
└── README.md 
```

## 2) How to Run

0. Python version: 3.11

1. Install dependencies:
py -m pip install -r requirements.txt

2. Run the project:
py src/logistic_regression.py

## 3) Machine Learning Model

The model uses features derived from transactional and behavioral data, including:

- Transaction amount
- Transaction hour
- Number of devices per user
- User transaction count
- User fraud rate (training-based)
- Merchant fraud rate (training-based)

To avoid data leakage, historical fraud rates were computed using only the training dataset.

The model outputs a **fraud probability (risk score)**, which is used in a decision engine.

## 4) Decision Engine

Transactions are classified into three categories:

- **Approve**: low-risk transactions  
- **Review**: medium-risk transactions  
- **Reject**: high-risk transactions  

Thresholds were adjusted to improve fraud detection recall, prioritizing financial risk reduction.

## 5) Outputs

The ROC curve and classification results are available in the `results/` folder.

- Accuracy: 94%  
- Recall (fraud): 66%  
- Precision (fraud): 86%  
- ROC-AUC: 0.89  

## 6) Report

A detailed analysis of the data, results, and conclusions is provided in the accompanying PDF report located in the 'report' folder.