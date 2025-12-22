import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dagshub.init(
    repo_owner="naufalwijdan14-ai",
    repo_name="Eksperimen_SML_Muhamad-Naufal-Wijdan",
    mlflow=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "preprocessing", "titanic_preprocessing.csv")

if not os.path.exists(data_path):
    data_path = "titanic_preprocessing.csv"

df = pd.read_csv(data_path)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Titanic_Baseline")

with mlflow.start_run(run_name="Baseline_Model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(acc)
