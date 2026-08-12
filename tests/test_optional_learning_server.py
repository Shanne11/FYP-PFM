import pytest

from server.optional_learning_server import LearningState, PARAMETER_COUNT


def payload(**overrides):
    value = {
        "update_type": "onnx_probability_calibration_delta",
        "base_model_contract": "pocketiq-class-weighted-proposed-seed42-v1",
        "base_calibration_version": 0,
        "delta": [0.01] * PARAMETER_COUNT,
        "sample_count": 10,
        "useful_note_count": 2,
        "mean_note_utility": 0.8,
    }
    value.update(overrides)
    return value


def test_rejects_wrong_parameter_count(tmp_path):
    state = LearningState(tmp_path, min_clients=2, token="x" * 24)
    with pytest.raises(ValueError, match="182"):
        state.accept(payload(delta=[0.1]))


def test_aggregates_with_bounded_utility_weights(tmp_path):
    state = LearningState(tmp_path, min_clients=2, token="x" * 24)
    state.accept(payload(mean_note_utility=1.0, sample_count=10))
    result = state.accept(
        payload(mean_note_utility=0.0, sample_count=10, delta=[-0.01] * PARAMETER_COUNT)
    )
    assert result["global_version"] == 1
    assert state.pending == []
    assert state.model["last_aggregation"]["clients"] == 2
    assert sum(state.model["last_aggregation"]["weights"]) == pytest.approx(1.0)


def test_missing_note_evidence_uses_neutral_multiplier(tmp_path):
    state = LearningState(tmp_path, min_clients=10, token="x" * 24)
    state.accept(payload(useful_note_count=0, mean_note_utility=0.99))
    assert state.pending[0]["utility_multiplier"] == 1.0


def test_rejects_stale_calibration_base_version(tmp_path):
    state = LearningState(tmp_path, min_clients=1, token="x" * 24)
    state.accept(payload())
    with pytest.raises(ValueError, match="stale calibration base version"):
        state.accept(payload(base_calibration_version=0))
