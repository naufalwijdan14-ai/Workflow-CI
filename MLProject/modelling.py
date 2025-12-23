import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


USER_NAME = "naufalwijdan14-ai"
REPO_NAME = "Eksperimen_SML_Muhamad-Naufal-Wijdan"
TOKEN = "49c11e904d62cef7fd7e9ef38b74edf94be2a4ee"

os.environ["MLFLOW_TRACKING_USERNAME"] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

mlflow.set_tracking_uri(f"https://dagshub.com/{USER_NAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("Titanic_Final_Project")

def train_advance_model():
    
    data_path = "titanic_preprocessing.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("preprocessing", "titanic_preprocessing.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Advance_Run_Naufal_Final"):

        n_est = 100
        max_d = 5
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan Hitung Metrik
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

   
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", max_d)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec) 
        mlflow.log_metric("recall", rec)      

   
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix Titanic")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") 
        plt.close()

      
        plt.figure(figsize=(8, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Feature Importances")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        
        mlflow.sklearn.log_model(model, "model_mlflow")
        
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

        print("-" * 30)
        print(f"STATUS: BERHASIL (LEVEL ADVANCE)")
        print(f"AKURASI: {acc}")
        print(f"ARTIFACTS: Folder model_mlflow, cm, & feature_importance tersimpan.")
        print("-" * 30)

if __name__ == "__main__":
    train_advance_model()
