import pandas as pd

# Load dataset
df = pd.read_csv("dataset/budgetwise_finance_dataset.csv")

print("Original Shape:", df.shape)

# Remove duplicate transactions
df = df.drop_duplicates()

# Remove rows without category
df = df.dropna(subset=["category"])

# Convert amount to numeric
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Remove rows without amount
df = df.dropna(subset=["amount"])

# Fill missing values
df["payment_mode"] = df["payment_mode"].fillna("Unknown")
df["location"] = df["location"].fillna("Unknown")
df["notes"] = df["notes"].fillna("")

print("Clean Shape:", df.shape)

print(df.head())

# Save cleaned dataset
df.to_csv("dataset/clean_budgetwise.csv", index=False)

print("Saved dataset/clean_budgetwise.csv")