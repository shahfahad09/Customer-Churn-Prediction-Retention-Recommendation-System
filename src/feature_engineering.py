import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


# ============================
# Paths
# ============================
DATA_PATH = r"D:\ML PROJECTS\CustomerChurnPrediction\Data\cleaned_customer_churn.csv"
MODEL_DIR = r"D:\ML PROJECTS\CustomerChurnPrediction\models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================
# Load Data
# ============================
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)


# ============================
# Separate Features and Target
# ============================
X = df.drop("Churn", axis=1)
y = df["Churn"]


# Encode target: No = 0, Yes = 1
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))


# ============================
# Identify Column Types
# ============================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)


# ============================
# Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ============================
# Preprocessing Pipeline
# ============================
categorical_pipeline = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_cols),
        ("numerical", numerical_pipeline, numerical_cols)
    ]
)


# ============================
# Fit and Transform Data
# ============================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed")
print("Processed Train Shape:", X_train_processed.shape)
print("Processed Test Shape:", X_test_processed.shape)


# ============================
# Save Processed Data
# ============================
joblib.dump(X_train_processed, os.path.join(MODEL_DIR, "X_train_processed.pkl"))
joblib.dump(X_test_processed, os.path.join(MODEL_DIR, "X_test_processed.pkl"))
joblib.dump(y_train, os.path.join(MODEL_DIR, "y_train.pkl"))
joblib.dump(y_test, os.path.join(MODEL_DIR, "y_test.pkl"))


# ============================
# Save Preprocessor
# ============================
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

print("\nAll files saved successfully in models folder.")