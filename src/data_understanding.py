import pandas as pd

# Display all columns
pd.set_option("display.max_columns", None)

# Load Dataset
df = pd.read_csv(r"D:\ML PROJECTS\CustomerChurnPrediction\Data\CustomerChurnData.csv")

# First 5 rows
print(df.head())

# Shape
print("\nShape of Dataset:")
print(df.shape)

# Information
print("\nDataset Information:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())


print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Variable Distribution:")
print(df["Churn"].value_counts())

print("\nTarget Variable Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\nUnique Values:")
print(df.nunique())