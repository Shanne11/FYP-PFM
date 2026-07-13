import os
import glob
import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models.mlp import MLP

from models.federated.client import FederatedClient
from models.federated.server import FederatedServer

from utils.federated_dataset import load_client_dataset

from utils.feature_engineering import (
    prepare_metadata_note_features,
    prepare_metadata_note_features_with_existing
)

# ==================================================
# Configuration
# ==================================================

OUTPUT = "outputs/proposed"

CLIENT_PATH = "data/clients/*.csv"

DATASET_PATH = "dataset/clean_budgetwise.csv"

ROUNDS = 10

LOCAL_EPOCHS = 3

MU = 0.01

os.makedirs(
    OUTPUT,
    exist_ok=True
)

print("=" * 60)
print("Proposed Model")
print("MLP + ACTM + Utility-weighted Aggregation")
print("=" * 60)

# ==================================================
# Load Client Files
# ==================================================

client_files = sorted(
    glob.glob(CLIENT_PATH)
)

print(f"Total Clients : {len(client_files)}")

if len(client_files) == 0:

    raise Exception(
        "No client datasets found. Run split_clients.py first."
    )

# ==================================================
# Build Global Feature Space
# ==================================================

print("\nPreparing global feature space...")

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

num_classes = len(
    encoders["category"].classes_
)

print(f"Input Features : {input_size}")

print(f"Output Classes : {num_classes}")

# ==================================================
# Global Model
# ==================================================

global_model = MLP(
    input_size,
    num_classes
)

server = FederatedServer()

# ==================================================
# Prepare Evaluation Dataset
# ==================================================

test_indices = pd.read_csv(
    "outputs/baseline2/test_indices.csv",
    header=None
)[0]

test_df = pd.read_csv(
    DATASET_PATH
)

test_df = test_df.iloc[
    test_indices
].reset_index(
    drop=True
)

X_test, y_test = prepare_metadata_note_features_with_existing(
    test_df,
    encoders,
    vectorizer
)

X_test = torch.FloatTensor(
    X_test.toarray()
)

y_test_tensor = torch.LongTensor(
    y_test.values.copy()
)

# ==================================================
# Federated Training
# ==================================================

round_results = []

best_accuracy = 0

print("\nStarting Federated Training...\n")

for current_round in range(ROUNDS):

    print(
        f"Communication Round "
        f"{current_round+1}/{ROUNDS}"
    )

    client_results = []

    global_weights = {

        name: param.detach().clone()

        for name, param in global_model.named_parameters()

    }

    # --------------------------------------------
    # Client Training
    # --------------------------------------------

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

        result = client.train(

            X_train,

            y_train,

            epochs=LOCAL_EPOCHS,

            global_weights=global_weights,

            mu=MU

        )

        client_results.append(
            result
        )

    # --------------------------------------------
    # Server Aggregation
    # --------------------------------------------

    new_weights, utilities = server.aggregate(
        client_results
    )

    global_model.load_state_dict(
        new_weights
    )

    print("\nClient Utility Scores")

    for i, utility in enumerate(utilities):

        print(
            f"Client {i+1:03d} : {utility:.4f}"
        )

    # --------------------------------------------
    # Evaluate Global Model
    # --------------------------------------------

    global_model.eval()

    with torch.no_grad():

        output = global_model(
            X_test
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction_numpy = prediction.numpy()

    actual_numpy = y_test_tensor.numpy()

    accuracy = accuracy_score(
        actual_numpy,
        prediction_numpy
    )

    print(
        f"Round Accuracy : {accuracy:.4f}"
    )

    # --------------------------------------------
    # Save Best Model
    # --------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(

            global_model.state_dict(),

            os.path.join(

                OUTPUT,

                "best_proposed_model.pth"

            )

        )

        print(
            "Best model updated."
        )

    # --------------------------------------------
    # Save Round Information
    # --------------------------------------------

    round_results.append(

        {

            "Round": current_round + 1,

            "Accuracy": accuracy,

            "Average Utility": float(sum(utilities) / len(utilities)),

            "Highest Utility": float(max(utilities)),

            "Lowest Utility": float(min(utilities))

        }

    )

print("\nTraining Completed.")

# ==================================================
# Final Evaluation
# ==================================================

print("\nRunning Final Evaluation...")

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
    f"Final Accuracy : {accuracy:.4f}"
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
        "Model : Proposed\n"
    )

    f.write(
        "Algorithm : MLP + ACTM + Utility-weighted Aggregation\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n\n"
    )

    f.write(
        report
    )

# ==================================================
# Save Utility Scores
# ==================================================

utility_df = pd.DataFrame(

    {

        "Client": list(

            range(

                1,

                len(utilities)+1

            )

        ),

        "Utility": utilities

    }

)

utility_df.to_csv(

    os.path.join(

        OUTPUT,

        "client_utilities.csv"

    ),

    index=False

)

# ==================================================
# Save Communication Rounds
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

    cm,

    cmap="viridis"

)

plt.title(

    "Proposed Model Confusion Matrix"

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
# Experiment Information
# ==================================================

with open(

    os.path.join(

        OUTPUT,

        "experiment_info.txt"

    ),

    "w"

) as f:

    f.write(
        "Experiment : Proposed Method\n"
    )

    f.write(
        "Model : Multi-Layer Perceptron\n"
    )

    f.write(
        "Aggregation : Utility-weighted Federated Aggregation\n"
    )

    f.write(
        "Trust Mechanism : ACTM\n"
    )

    f.write(
        f"Communication Rounds : {ROUNDS}\n"
    )

    f.write(
        f"Local Epochs : {LOCAL_EPOCHS}\n"
    )

    f.write(
        f"FedProx Mu : {MU}\n"
    )

    f.write(
        f"Clients : {len(client_files)}\n"
    )

    f.write(
        f"Input Features : {input_size}\n"
    )

    f.write(
        f"Output Classes : {num_classes}\n"
    )

    f.write(
        f"Best Accuracy : {best_accuracy:.4f}\n"
    )

print("\n==========================================")
print(" Proposed Model Training Completed ")
print("==========================================")