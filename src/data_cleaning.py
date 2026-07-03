import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv(r"D:\ML PROJECTS\CustomerChurnPrediction\Data\CustomerChurnData.csv")

print("Before Cleaning:")
print(df.info())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing Values After TotalCharges Conversion:")
print(df.isnull().sum())

# Drop rows where TotalCharges is missing
df.dropna(subset=["TotalCharges"], inplace=True)

# Drop customerID because it is only an identifier
df.drop("customerID", axis=1, inplace=True)

print("\nAfter Cleaning:")
print(df.info())

print("\nFinal Shape:")
print(df.shape)

# Save cleaned data
df.to_csv(r"D:\ML PROJECTS\CustomerChurnPrediction\Data\cleaned_customer_churn.csv", index=False)

print("\nCleaned data saved successfully.")