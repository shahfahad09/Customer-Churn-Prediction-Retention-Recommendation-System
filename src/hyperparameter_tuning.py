import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

MODEL_DIR = r"D:\ML PROJECTS\CustomerChurnPrediction\models"
RESULTS_DIR = r"D:\ML PROJECTS\CustomerChurnPrediction\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = joblib.load(os.path.join(MODEL_DIR, "X_train_processed.pkl"))
X_test = joblib.load(os.path.join(MODEL_DIR, "X_test_processed.pkl"))
y_train = joblib.load(os.path.join(MODEL_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))

print("Data loaded successfully")

model = LogisticRegression(max_iter=1000, random_state=42)

params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

results = {
    "Model": "Tuned Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
}

print("\nTuned Model Performance:")
for key, value in results.items():
    print(key, ":", round(value, 4) if isinstance(value, float) else value)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

pd.DataFrame([results]).to_csv(
    os.path.join(RESULTS_DIR, "tuned_logistic_regression_results.csv"),
    index=False
)

joblib.dump(best_model, os.path.join(MODEL_DIR, "tuned_logistic_regression.pkl"))

print("\nTuned Logistic Regression saved successfully.")