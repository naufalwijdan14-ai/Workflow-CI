import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi DagsHub
token = os.getenv("DAGSHUB_USER_TOKEN")

dagshub.init(
    repo_owner="naufalwijdan14-ai",
    repo_name="Eksperimen_SML_Muhamad-Naufal-Wijdan",
    mlflow=True
)

if token:
    mlflow.set_tracking_uri(f"https://dagshub.com/naufalwijdan14-ai/Eksperimen_SML_Muhamad-Naufal-Wijdan.mlflow")

# Pengaturan Path Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Mencoba mencari file di folder preprocessing
data_path = os.path.join(BASE_DIR, "..", "preprocessing", "titanic_preprocessing.csv")

if not os.path.exists(data_path):
    # Jika gagal, coba cari di direktori yang sama
    data_path = os.path.join(BASE_DIR, "titanic_preprocessing.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File data tidak ditemukan di: {data_path}")

# Load Data
df = pd.read_csv(data_path)
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow Experiment
mlflow.set_experiment("Titanic_Baseline")

with mlflow.start_run(run_name="Baseline_Model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log Parameter & Metric
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    
    # Log Model - ini akan membuat folder 'model' di dalam 'mlruns'
    mlflow.sklearn.log_model(model, "model")

    print(f"Model Training Selesai. Accuracy: {acc}")
