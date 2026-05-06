import json
import joblib
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")

FEATURES = [
    "TransactionAmt", "ProductCD", "card4", "card6",
    "P_emaildomain", "TransactionDT"
]

def load_model():
    print("Loading model and encoders...")
    model = joblib.load("model/fraud_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    print("Model loaded successfully")
    return model, encoders

def get_db_engine():
    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    return create_engine(url)

def preprocess_transaction(transaction, encoders):
    cat_cols = {
        "ProductCD": "ProductCD",
        "card4": "card4",
        "card6": "card6",
        "P_emaildomain": "P_emaildomain"
    }

    row = {}
    for feature in FEATURES:
        row[feature] = transaction.get(feature)

    for col, enc_key in cat_cols.items():
        val = str(row.get(col) or "unknown")
        le = encoders[enc_key]
        if val in le.classes_:
            row[col] = le.transform([val])[0]
        else:
            row[col] = 0

    for col in ["TransactionAmt", "TransactionDT"]:
        try:
            row[col] = float(row[col]) if row[col] is not None else 0.0
        except:
            row[col] = 0.0

    return pd.DataFrame([row])[FEATURES]

def save_to_db(engine, transaction, prediction, probability):
    query = text("""
        INSERT INTO transactions (
            transaction_id, transaction_amt, product_cd,
            card4, card6, p_emaildomain, transaction_dt,
            is_fraud_actual, is_fraud_predicted, fraud_probability
        ) VALUES (
            :transaction_id, :transaction_amt, :product_cd,
            :card4, :card6, :p_emaildomain, :transaction_dt,
            :is_fraud_actual, :is_fraud_predicted, :fraud_probability
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)

    with engine.connect() as conn:
        conn.execute(query, {
            "transaction_id": str(transaction.get("TransactionID", "")),
            "transaction_amt": float(transaction.get("TransactionAmt") or 0),
            "product_cd": str(transaction.get("ProductCD") or ""),
            "card4": str(transaction.get("card4") or ""),
            "card6": str(transaction.get("card6") or ""),
            "p_emaildomain": str(transaction.get("P_emaildomain") or ""),
            "transaction_dt": int(transaction.get("TransactionDT") or 0),
            "is_fraud_actual": int(transaction.get("isFraud") or 0),
            "is_fraud_predicted": int(prediction),
            "fraud_probability": float(round(probability, 4))
        })
        conn.commit()

def process_stream():
    model, encoders = load_model()
    engine = get_db_engine()

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fraud-processor",
        consumer_timeout_ms=10000
    )

    print(f"Listening to Kafka topic: {KAFKA_TOPIC}")
    print("Processing transactions...\n")

    total = 0
    fraud_count = 0

    try:
        for message in consumer:
            transaction = message.value

            features_df = preprocess_transaction(transaction, encoders)
            probability = model.predict_proba(features_df)[0][1]
            prediction = 1 if probability >= 0.5 else 0

            save_to_db(engine, transaction, prediction, probability)

            total += 1
            if prediction == 1:
                fraud_count += 1
                amt = transaction.get("TransactionAmt", 0)
                txid = transaction.get("TransactionID", "")
                print(f"FRAUD DETECTED | TxID: {txid} | Amount: ${amt:.2f} | Probability: {probability:.2%}")

            if total % 100 == 0:
                print(f"Processed {total:,} transactions | Fraud detected: {fraud_count}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        consumer.close()
        print(f"\nDone. Total processed: {total:,} | Fraud flagged: {fraud_count}")

if __name__ == "__main__":
    process_stream()
