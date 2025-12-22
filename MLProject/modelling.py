import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "titanic_preprocessing.csv")

df = pd.read_csv(data_path)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="CI_Automated_Run"):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")

    print(f"Accuracy: {acc}")
