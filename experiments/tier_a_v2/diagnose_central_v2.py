
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare_features_v2 import V2FeatureBuilder
from models.mlp import MLP
from utils.experiment_data import load_experiment_data
from utils.class_balance import effective_number_weights

torch.manual_seed(42)
np.random.seed(42)

train, validation, _, manifest = load_experiment_data(
    str(V2 / "data/clean_budgetwise_v2.csv"),
    str(V2 / "data/experiment_split_v2.json"),
)

assert manifest["group_key"] == "transaction_id"

builder = V2FeatureBuilder().fit(train)

def prepare(frame):
    meta, notes, _, y = builder.transform_parts(frame)
    x = hstack([meta, notes], format="csr").toarray()
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

train_x, train_y = prepare(train)
val_x, val_y = prepare(validation)

assert train_x.shape[1] == val_x.shape[1] == 108

weights = effective_number_weights(
    train_y.numpy(), 13, beta=0.9999
)

model = MLP(108, 13)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=32,
    shuffle=True,
)

output = V2 / "outputs/diagnostics/centralised_seed_42"
output.mkdir(parents=True, exist_ok=True)

results = []
best_f1 = -1.0
best_epoch = None

for epoch in range(1, 31):
    model.train()

    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(train_x).argmax(axis=1).numpy()
        logits = model(val_x)
        val_pred = logits.argmax(axis=1).numpy()
        val_loss = criterion(logits, val_y).item()

    actual = val_y.numpy()
    recalls = recall_score(
        actual, val_pred,
        labels=np.arange(13),
        average=None,
        zero_division=0
    )

    row = {
        "epoch": epoch,
        "train_macro_f1": f1_score(
            train_y.numpy(), train_pred,
            average="macro", zero_division=0
        ),
        "validation_macro_f1": f1_score(
            actual, val_pred,
            average="macro", zero_division=0
        ),
        "validation_accuracy": accuracy_score(actual, val_pred),
        "validation_loss": val_loss,
        "predicted_categories": int(len(np.unique(val_pred))),
        "zero_recall_categories": int(np.sum(recalls == 0)),
    }

    results.append(row)

    if row["validation_macro_f1"] > best_f1:
        best_f1 = row["validation_macro_f1"]
        best_epoch = epoch
        torch.save(model.state_dict(), output / "best_model.pt")

    print(
        f"Epoch {epoch:02d}: "
        f"train={row['train_macro_f1']:.4f}, "
        f"validation={row['validation_macro_f1']:.4f}, "
        f"coverage={row['predicted_categories']}/13",
        flush=True,
    )

pd.DataFrame(results).to_csv(
    output / "epoch_metrics.csv", index=False
)

summary = {
    "experiment": "V2 centralised diagnostic",
    "seed": 42,
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_f1,
    "best_epoch_metrics": results[best_epoch - 1],
    "features": 108,
    "loss": "effective_number_beta_0.9999",
    "test_rows_scored": 0,
}

(output / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\nBEST VALIDATION MACRO F1:", round(best_f1, 5))
print("BEST EPOCH:", best_epoch)
print("Saved:", output)
