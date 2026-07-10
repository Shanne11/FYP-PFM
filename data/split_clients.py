import os
import pandas as pd

OUTPUT = "data/clients"

os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv("dataset/clean_budgetwise.csv")

# Split by user_id
groups = df.groupby("user_id")

print(f"Total Clients : {len(groups)}")

for user_id, client_df in groups:

    client_df.to_csv(
        os.path.join(
            OUTPUT,
            f"{user_id}.csv"
        ),
        index=False
    )

print("Client datasets created.")