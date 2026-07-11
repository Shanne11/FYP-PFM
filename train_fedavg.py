import os
import glob
import copy
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


from models.mlp import MLP

from federated.client import FederatedClient
from federated.server import FederatedServer

from utils.federated_dataset import load_client_dataset
from utils.feature_engineering import prepare_metadata_note_features


# ==================================================
# Configuration
# ==================================================

OUTPUT = "outputs/baseline4"

CLIENT_PATH = "data/clients/*.csv"

DATASET_PATH = "dataset/clean_budgetwise.csv"


ROUNDS = 10

LOCAL_EPOCHS = 3


os.makedirs(
    OUTPUT,
    exist_ok=True
)


print("=" * 60)
print("Baseline 4 - Federated Learning FedAvg")
print("=" * 60)



# ==================================================
# Load Clients
# ==================================================

print("\nLoading clients...")


client_files = sorted(
    glob.glob(CLIENT_PATH)
)


print(
    f"Total Clients : {len(client_files)}"
)


if len(client_files) == 0:

    raise Exception(
        "No client dataset found. Run split_clients.py first."
    )



# ==================================================
# Determine Input Size
# ==================================================

print("\nPreparing model information...")

full_df = pd.read_csv(
    DATASET_PATH
)

_, _, encoders, vectorizer = prepare_metadata_note_features(
    full_df
)

sample_X, _ = load_client_dataset(
    client_files[0],
    encoders,
    vectorizer
)

input_size = sample_X.shape[1]

# FIX: Get the true total number of classes from the global label encoder
num_classes = len(encoders["category"].classes_)


print(
    f"Input Features : {input_size}"
)

print(
    f"Classes (Global): {num_classes}" 
)



# ==================================================
# Create Global Model
# ==================================================

global_model = MLP(
    input_size,
    num_classes
)



server = FederatedServer()



# ==================================================
# Federated Training
# ==================================================

print("\nStarting Federated Training...")


round_results = []


for current_round in range(ROUNDS):


    print(
        f"\nCommunication Round "
        f"{current_round + 1}/{ROUNDS}"
    )


    client_weights = []



    # -------------------------------
    # Client Training
    # -------------------------------

    for index, client_file in enumerate(client_files):


        print(
            f" Client {index+1}/{len(client_files)}"
        )


        X_train, y_train = load_client_dataset(
            client_file,
            encoders,
            vectorizer
        )


        local_model = copy.deepcopy(
            global_model
        )


        client = FederatedClient(
            local_model
        )


        weights = client.train(
            X_train,
            y_train,
            epochs=LOCAL_EPOCHS
        )


        client_weights.append(
            weights
        )



    # -------------------------------
    # FedAvg Aggregation
    # -------------------------------


    new_weights = server.aggregate(
        client_weights
    )


    global_model.load_state_dict(
        new_weights
    )


    round_results.append(
        {
            "Round": current_round + 1,
            "Clients": len(client_files)
        }
    )


    print(
        "Aggregation completed"
    )



# ==================================================
# Save Global Model
# ==================================================

model_path = os.path.join(
    OUTPUT,
    "fedavg_model.pth"
)


torch.save(
    global_model.state_dict(),
    model_path
)


print(
    "\nGlobal model saved."
)



# ==================================================
# Evaluation
# ==================================================

print("\nEvaluating Global Model...")


df = pd.read_csv(
    DATASET_PATH
)

from utils.feature_engineering import prepare_metadata_note_features_with_existing


X_test, y_test, _, _ = prepare_metadata_note_features(
    df
)


X_test = torch.FloatTensor(
    X_test.toarray()
)


y_test_tensor = torch.LongTensor(y_test.values.copy())

global_model.eval()


with torch.no_grad():

    output = global_model(
        X_test
    )

    prediction = torch.argmax(
        output,
        dim=1
    )



prediction = prediction.numpy()

actual = y_test_tensor.numpy()



accuracy = accuracy_score(
    actual,
    prediction
)



print(
    f"Accuracy : {accuracy:.4f}"
)



# ==================================================
# Save Predictions
# ==================================================

prediction_df = pd.DataFrame(
    {
        "Actual": actual,
        "Predicted": prediction
    }
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT,
        "predictions.csv"
    ),
    index=False
)



# ==================================================
# Save Metrics
# ==================================================

report = classification_report(
    actual,
    prediction,
    zero_division=0
)



with open(
    os.path.join(
        OUTPUT,
        "metrics.txt"
    ),
    "w"
) as f:


    f.write(
        f"Accuracy: {accuracy:.4f}\n\n"
    )

    f.write(
        report
    )



# ==================================================
# Confusion Matrix
# ==================================================

cm = confusion_matrix(
    actual,
    prediction
)


plt.figure(
    figsize=(8,6)
)


plt.imshow(
    cm
)


plt.title(
    "FedAvg Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.colorbar()


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT,
        "confusion_matrix.png"
    )
)


plt.close()



# ==================================================
# Save Round Information
# ==================================================

pd.DataFrame(
    round_results
).to_csv(
    os.path.join(
        OUTPUT,
        "communication_rounds.csv"
    ),
    index=False
)



with open(
    os.path.join(
        OUTPUT,
        "experiment_info.txt"
    ),
    "w"
) as f:


    f.write(
        "Baseline: FedAvg\n"
    )

    f.write(
        "Model: MLP\n"
    )

    f.write(
        f"Rounds: {ROUNDS}\n"
    )

    f.write(
        f"Local Epochs: {LOCAL_EPOCHS}\n"
    )

    f.write(
        f"Clients: {len(client_files)}\n"
    )



print("\n================================")
print("Baseline 4 FedAvg Completed")
print("================================")