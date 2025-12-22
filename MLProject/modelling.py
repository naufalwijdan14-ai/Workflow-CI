import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

os.environ['DAGSHUB_NON_INTERACTIVE'] = '1'

dagshub.init(
    repo_owner='naufalwijdan14-ai', 
    repo_name='Eksperimen_SML_Muhamad-Naufal-Wijdan', 
    mlflow=True
)

dagshub.init(repo_owner='naufalwijdan14-ai', 
             repo_name='Eksperimen_SML_Muhamad-Naufal-Wijdan', 
             mlflow=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "titanic_preprocessing.csv")

if not os.path.exists(data_path):
   
    data_path = "titanic_preprocessing.csv" 

df = pd.read_csv(data_path)
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  MLflow Tracking
mlflow.set_experiment("Titanic_CI_Workflow")

with mlflow.start_run(run_name="CI_Automated_Run"):
    # Gunakan parameter yang dikirim dari MLProject/GitHub Actions
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Logging
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Berhasil! Akurasi CI: {acc}")
