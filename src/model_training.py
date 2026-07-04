import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

for model_name, model in models.items():
    print("\n" + "=" * 50)
    print(f"Training: {model_name}")
    print("=" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC-AUC", ascending=False)

results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 50)
print("Model Comparison")
print("=" * 50)
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

print(f"\nBest Model: {best_model_name}")
print("Best model saved as best_model.pkl")
print(f"Results saved at: {results_path}")
