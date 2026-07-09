import pandas as pd

from models.rules import predict

# Load cleaned dataset
df = pd.read_csv("dataset/clean_budgetwise.csv")

correct = 0
total = 0

for _, row in df.iterrows():

    prediction = predict(row["notes"])

    actual = row["category"]

    if prediction == actual:

        correct += 1

    total += 1

accuracy = correct / total

print(f"Total Transactions : {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy           : {accuracy:.4f}")