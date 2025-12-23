import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

USER_NAME = "naufalwijdan14-ai"
REPO_NAME = "Eksperimen_SML_Muhamad-Naufal-Wijdan"
TOKEN = "49c11e904d62cef7fd7e9ef38b74edf94be2a4ee"

os.environ["MLFLOW_TRACKING_USERNAME"] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

mlflow.set_tracking_uri(f"https://dagshub.com/{USER_NAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("Titanic_Final_Project")

# Autolog memenuhi syarat Advanced dengan menghasilkan >2 artefak (CM, ROC, dll)
mlflow.sklearn.autolog(log_models=True)

def train_model():

    data_path = "MLProject/titanic_preprocessing.csv" if os.path.exists("MLProject") else "titanic_preprocessing.csv"

    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Run_Kriteria_3_Retraining"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan pkl untuk kebutuhan workflow Docker/Push
        joblib.dump(model, "model.pkl")
        
        print("Training di GitHub Actions Selesai!")

if __name__ == "__main__":
    train_model()
