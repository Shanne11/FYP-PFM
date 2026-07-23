"""Validate the frozen research model contract against repository metadata."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.experiment_data import CATEGORY_ALIASES


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="deployment/research_model_contract.json")
    parser.add_argument("--manifest", default="data/experiment_split.json")
    return parser.parse_args()


def validate(contract, manifest):
    errors = []
    categories = contract["categories"]
    canonical = sorted(set(CATEGORY_ALIASES.values()))
    if categories != canonical:
        errors.append("Contract category order does not match the canonical label order")
    if contract["model"]["output_classes"] != len(categories):
        errors.append("Output class count does not match category count")
    if contract["model"]["input_features"] != (
        contract["model"]["metadata_features"] + contract["model"]["note_features"]
    ):
        errors.append("Input feature dimensions do not add up")
    if contract["model"]["note_features"] != len(contract["preprocessing"]["note_vocabulary"]):
        errors.append("Note feature count does not match vocabulary size")
    if contract["dataset"]["fingerprint"] != manifest["dataset_fingerprint"]:
        errors.append("Dataset fingerprint differs from frozen split manifest")
    if contract["dataset"]["split_manifest_version"] != manifest["version"]:
        errors.append("Split manifest version differs from contract")
    if contract["seed_policy"]["deployment_checkpoint_seed"] not in contract["seed_policy"]["research_seeds"]:
        errors.append("Deployment seed is not one of the evaluated research seeds")
    if errors:
        raise ValueError("; ".join(errors))
    return True


def main(config):
    contract = json.loads(Path(config.contract).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config.manifest).read_text(encoding="utf-8"))
    validate(contract, manifest)
    print("Frozen research model contract is internally consistent.")


if __name__ == "__main__":
    main(arguments())
