# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import time
from prometheus_client import start_http_server, Gauge

# Inisialisasi Metrik Prometheus (Untuk kriteria Advanced Monitoring)
accuracy_gauge = Gauge('model_accuracy_score', 'Akurasi dari model yang dilatih')

def train_model():
    # Parsing argumen agar bisa menerima input path data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"File tidak ditemukan: {args.data_path}")
        
    data = pd.read_csv(args.data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Preprocessing sederhana (Label Encoding untuk kolom teks)
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  MLflow Tracking
    with mlflow.start_run(run_name="Titanic_Advanced_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Hasil Akurasi: {acc}")
        
        # Update Metrik Prometheus
        accuracy_gauge.set(acc)

        # Log ke MLflow Dashboard
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan file model secara fisik untuk Artifact GitHub
        joblib.dump(model, "model.pkl")
        print("Model tersimpan sebagai model.pkl")

if __name__ == "__main__":
    # Jalankan server Prometheus di port 8000
    start_http_server(8000)
    print("Prometheus metrics server berjalan di port 8000")
    
    train_model()
    
    # Memberi waktu server Prometheus agar bisa dibaca (opsional di CI)
    print("Sinkronisasi Prometheus (5 detik)...")
    time.sleep(5)
