import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

FEATURES = [
    "TransactionAmt", "ProductCD", "card4", "card6",
    "P_emaildomain", "TransactionDT"
]
TARGET = "isFraud"

def load_data():
    print("Loading data...")
    df = pd.read_csv(
        "data/train_transaction.csv",
        nrows=50000,
        usecols=FEATURES + [TARGET, "TransactionID"]
    )
    print(f"Loaded {len(df):,} rows")
    return df

def preprocess(df):
    print("Preprocessing...")
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain"]
    encoders = {}

    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    for col in ["TransactionAmt", "TransactionDT"]:
        df[col] = df[col].fillna(df[col].median())

    return df, encoders

def train(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train):,} samples...")
    print(f"Fraud rate: {y_train.mean():.2%}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    return model, X_test

def save_artifacts(model, encoders, X_test):
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/fraud_model.pkl")
    joblib.dump(encoders, "model/encoders.pkl")
    print("\nModel saved to model/fraud_model.pkl")

    print("Computing SHAP values (this takes ~30 seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.iloc[:200]
    shap_values = explainer.shap_values(shap_sample)
    joblib.dump({"explainer": explainer, "sample": shap_sample, "values": shap_values}, "model/shap_data.pkl")
    print("SHAP data saved to model/shap_data.pkl")

def main():
    df = load_data()
    df, encoders = preprocess(df)
    model, X_test = train(df)
    save_artifacts(model, encoders, X_test)
    print("\nPhase 3 complete — model ready")

if __name__ == "__main__":
    main()
