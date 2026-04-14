import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from functions import roc_curve_visualization

df = pd.read_csv("../data/transactional-sample.csv")

# Feature Engineering

# Date to Datetime Type
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Creating the Hour Feature
df["hour"] = df["transaction_date"].dt.hour

# Creating the Number of Devices Per User
df["num_devices_per_user"] = df.groupby("user_id")["device_id"].transform("nunique")

# Creating the Number of Transactions Per User
df["user_transaction_count"] = df.groupby("user_id")["transaction_id"].transform("count")

# Custom Transformer for target-based features

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.user_fraud_rate = None
        self.merchant_fraud_rate = None

    def fit(self, X, y):
        df_temp = X.copy()
        df_temp["has_cbk"] = y

        self.user_fraud_rate = df_temp.groupby("user_id")["has_cbk"].mean()
        self.merchant_fraud_rate = df_temp.groupby("merchant_id")["has_cbk"].mean()

        return self

    def transform(self, X):
        X = X.copy()

        X["user_fraud_rate"] = X["user_id"].map(self.user_fraud_rate)
        X["merchant_fraud_rate"] = X["merchant_id"].map(self.merchant_fraud_rate)

        X["user_fraud_rate"] = X["user_fraud_rate"].fillna(0)
        X["merchant_fraud_rate"] = X["merchant_fraud_rate"].fillna(0)

        return X

# Separating the Data in Train and Test:

X = df[[
    "user_id",
    "merchant_id",
    "transaction_amount",
    "hour",
    "num_devices_per_user",
    "user_transaction_count"
]].copy()

y = df["has_cbk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

features = [
    "transaction_amount",
    "hour",
    "num_devices_per_user",
    "user_transaction_count",
    "user_fraud_rate",
    "merchant_fraud_rate"
]

# Creating and Training the Model

pipeline = Pipeline([
    ("target_encoder", TargetEncoder()),
    ("model", LogisticRegression(max_iter=1000))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Transform data
X_train_transformed = pipeline.named_steps["target_encoder"].transform(X_train)
X_test_transformed = pipeline.named_steps["target_encoder"].transform(X_test)

# Select final features
X_train_final = X_train_transformed[features]
X_test_final = X_test_transformed[features]

# Train model
pipeline.named_steps["model"].fit(X_train_final, y_train)

# Predict
y_pred = pipeline.named_steps["model"].predict(X_test_final)
y_prob = pipeline.named_steps["model"].predict_proba(X_test_final)[:, 1]

# Evaluating the Model With Different Thresholds:

#for t in [0.2, 0.3, 0.4, 0.5]:
#    y_pred_custom = (y_prob > t).astype(int)
#    print(f"\nThreshold: {t}")
#    print(classification_report(y_test, y_pred_custom))

# Evaluating the Best Model:

threshold = 0.2

y_pred_custom = (y_prob > threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

roc_curve_visualization(y_test, y_prob, show=True)

# Decision Engine

def decision(score):
    if score > 0.7:
        return "Reject"
    elif score > 0.2:
        return "Review"
    else:
        return "Approve"

decisions = [decision(s) for s in y_prob]

print(pd.Series(decisions).value_counts())