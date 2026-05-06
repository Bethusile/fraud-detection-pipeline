# Fraud Detection Pipeline
Real-time fraud detection pipeline using Kafka, PySpark, XGBoost and Streamlit.

## Overview
End-to-end data engineering and ML project simulating a live transaction stream,
scoring transactions for fraud in real time, storing results in PostgreSQL,
and visualising alerts on a Streamlit dashboard.

---

## Tech stack

| Layer | Technology |
|---|---|
| Message streaming | Apache Kafka |
| Stream processing | Python consumer with real-time scoring |
| ML model | XGBoost (ROC-AUC: 0.83) |
| Explainability | SHAP values |
| Database | PostgreSQL |
| Dashboard | Streamlit + Plotly |
| Infrastructure | Docker + Docker Compose |
| Dataset | IEEE-CIS Fraud Detection (Kaggle, 590k transactions) |

---

## Features

- Real-time transaction streaming via Kafka producer simulating a live payments feed
- XGBoost fraud classifier trained on 50,000 transactions with class imbalance handling
- SHAP explainability layer showing which features drive each fraud prediction
- PostgreSQL storage with indexed queries for fast dashboard retrieval
- Live Streamlit dashboard with fraud metrics, probability distribution, and flagged transaction table
- Adjustable fraud probability threshold slider for real-time alert tuning
- Dockerised infrastructure — spins up with a single command

---

## Project structure
fraud-detection-pipeline/
├── docker-compose.yml        # Kafka + PostgreSQL infrastructure
├── requirements.txt
├── producer/
│   └── kafka_producer.py     # Streams transactions into Kafka
├── processor/
│   └── stream_processor.py   # Consumes stream, scores with ML model, writes to DB
├── model/
│   └── train_model.py        # XGBoost training + SHAP export
├── database/
│   └── schema.sql            # PostgreSQL schema
└── dashboard/
└── app.py                # Streamlit dashboard

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Bethusile/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start infrastructure
```bash
docker-compose up -d
```

### 4. Set up the database
```bash
docker exec -i <postgres_container> psql -U fraud_user -d fraud_db < database/schema.sql
```

### 5. Train the model
```bash
python model/train_model.py
```

### 6. Run the pipeline (two terminals)
```bash
# Terminal 1
python producer/kafka_producer.py

# Terminal 2
python processor/stream_processor.py
```

### 7. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Model performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.83 |
| Fraud recall | 66% |
| Fraud precision | 12% |
| Training samples | 40,000 |
| Fraud rate in training data | 2.71% |

Recall is prioritised over precision — in fraud detection, missing a fraudulent transaction is more costly than a false positive.

---

## What I would improve with more time

- Add more features from the identity table to improve recall
- Implement model retraining pipeline as new labelled data arrives
- Add data quality checks and alerting on schema drift
- Deploy dashboard to a cloud platform (Render or Railway)
- Add unit tests for the producer and processor

---

## Dataset

IEEE-CIS Fraud Detection — Kaggle competition dataset  
https://www.kaggle.com/c/ieee-fraud-detection
