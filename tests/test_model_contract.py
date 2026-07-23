import json
from pathlib import Path

from evaluation.validate_model_contract import validate


def test_frozen_model_contract_matches_repository_manifest():
    contract = json.loads(Path("deployment/research_model_contract.json").read_text())
    manifest = json.loads(Path("data/experiment_split.json").read_text())
    assert validate(contract, manifest)
