import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Paths

DATA_PATH = r"D:\ML PROJECTS\CustomerChurnPrediction\Data\cleaned_customer_churn.csv"
IMAGE_DIR = r"D:\ML PROJECTS\CustomerChurnPrediction\images\eda_graphs"


# Prepare Image Folder
if os.path.exists(IMAGE_DIR):
    shutil.rmtree(IMAGE_DIR)

os.makedirs(IMAGE_DIR, exist_ok=True)

# Load Dataset
df = pd.read_csv(DATA_PATH)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

def save_plot(filename):
    path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_count(feature, title, filename, figsize=(8, 5), rotation=0):
    plt.figure(figsize=figsize)

    ax = sns.countplot(
        data=df,
        x=feature,
        hue="Churn"
    )

    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=rotation)

    for container in ax.containers:
        ax.bar_label(container)

    save_plot(filename)

    print(f"\n{title} (%)")
    print(pd.crosstab(df[feature], df["Churn"], normalize="index") * 100)


def plot_hist(feature, title, filename, xlabel):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue="Churn",
        bins=30,
        kde=True,
        multiple="stack"
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Customers")

    save_plot(filename)


# Target Variable Analysis
plt.figure(figsize=(6, 5))

ax = sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

for container in ax.containers:
    ax.bar_label(container)

save_plot("01_churn_distribution.png")

print("\nChurn Distribution (%)")
print(df["Churn"].value_counts(normalize=True) * 100)


# Pie Chart
plt.figure(figsize=(6, 6))

df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Churn Percentage")
plt.ylabel("")

save_plot("02_churn_percentage.png")


# 2. Categorical Feature Analysis
plot_count("gender", "Gender vs Customer Churn", "03_gender_vs_churn.png", figsize=(6, 5))

plot_count("SeniorCitizen", "Senior Citizen vs Customer Churn", "04_senior_citizen_vs_churn.png", figsize=(6, 5))

plot_count("Partner", "Partner vs Customer Churn", "05_partner_vs_churn.png", figsize=(6, 5))

plot_count("Dependents", "Dependents vs Customer Churn", "06_dependents_vs_churn.png", figsize=(6, 5))

plot_count("Contract", "Contract Type vs Customer Churn", "07_contract_vs_churn.png", figsize=(8, 5))

plot_count("InternetService", "Internet Service vs Customer Churn", "08_internet_service_vs_churn.png", figsize=(8, 5))

plot_count("OnlineSecurity", "Online Security vs Customer Churn", "09_online_security_vs_churn.png", figsize=(8, 5))

plot_count("TechSupport", "Tech Support vs Customer Churn", "10_tech_support_vs_churn.png", figsize=(8, 5))

plot_count("PaymentMethod", "Payment Method vs Customer Churn", "11_payment_method_vs_churn.png", figsize=(11, 5), rotation=15)

plot_count("PaperlessBilling", "Paperless Billing vs Customer Churn", "12_paperless_billing_vs_churn.png", figsize=(6, 5))



plot_hist(
    "tenure",
    "Tenure Distribution by Churn",
    "13_tenure_distribution_by_churn.png",
    "Tenure (Months)"
)

plot_hist(
    "MonthlyCharges",
    "Monthly Charges Distribution by Churn",
    "14_monthly_charges_distribution_by_churn.png",
    "Monthly Charges"
)

plot_hist(
    "TotalCharges",
    "Total Charges Distribution by Churn",
    "15_total_charges_distribution_by_churn.png",
    "Total Charges"
)


print("\nEDA completed successfully.")
print(f"All graphs saved in: {IMAGE_DIR}")
