"""Authenticated localhost server for deployment calibration experiments.

It accepts only a clipped 182-number calibration delta plus aggregate counts.
It must not be described as the Chapter 5 full-MLP federated simulation.
"""

import argparse
import hmac
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PARAMETER_COUNT = 182
MODEL_CONTRACT = "pocketiq-class-weighted-proposed-seed42-v1"


def clip(value, lower, upper):
    return max(lower, min(upper, value))


class LearningState:
    def __init__(self, directory, min_clients, token):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.min_clients = min_clients
        self.token = token
        self.model_path = self.directory / "global_calibration.json"
        self.pending_path = self.directory / "pending_updates.json"
        self.model = self._read(self.model_path, {"version": 0, "parameters": [0.0] * PARAMETER_COUNT})
        self.pending = self._read(self.pending_path, [])

    @staticmethod
    def _read(path, fallback):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback
        except (OSError, ValueError):
            return fallback

    @staticmethod
    def _write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)

    def accept(self, payload):
        if payload.get("update_type") != "onnx_probability_calibration_delta":
            raise ValueError("unsupported update type")
        if payload.get("base_model_contract") != MODEL_CONTRACT:
            raise ValueError("base model contract mismatch")
        base_version = int(payload.get("base_calibration_version", -1))
        current_version = int(self.model.get("version", 0))
        if base_version != current_version:
            raise ValueError(
                f"stale calibration base version: expected {current_version}, got {base_version}"
            )
        if self.pending and self.pending[0]["base_calibration_version"] != base_version:
            raise ValueError("pending updates do not share one calibration base version")
        delta = payload.get("delta")
        if not isinstance(delta, list) or len(delta) != PARAMETER_COUNT:
            raise ValueError(f"delta must contain {PARAMETER_COUNT} parameters")
        delta = [float(value) for value in delta]
        if not all(math.isfinite(value) for value in delta):
            raise ValueError("delta contains non-finite values")
        samples = int(payload.get("sample_count", 0))
        if samples <= 0:
            raise ValueError("sample_count must be positive")
        norm = math.sqrt(sum(value * value for value in delta))
        if norm > 1.0:
            delta = [value / norm for value in delta]
            norm = 1.0
        utility = float(payload.get("mean_note_utility", 0.0))
        note_count = int(payload.get("useful_note_count", 0))
        valid_utility = note_count > 0 and math.isfinite(utility)
        multiplier = clip(0.75 + 0.5 * clip(utility, 0.0, 1.0), 0.75, 1.25) if valid_utility else 1.0
        self.pending.append(
            {
                "delta": delta,
                "sample_count": samples,
                "utility_multiplier": multiplier,
                "update_norm": norm,
                "base_calibration_version": base_version,
            }
        )
        self._write(self.pending_path, self.pending)
        if len(self.pending) >= self.min_clients:
            self.aggregate()
        return {"accepted": True, "pending_clients": len(self.pending), "global_version": self.model["version"]}

    def aggregate(self):
        if not self.pending:
            return None
        raw_weights = [item["sample_count"] * item["utility_multiplier"] for item in self.pending]
        denominator = sum(raw_weights)
        coefficients = [value / denominator for value in raw_weights]
        change = [0.0] * PARAMETER_COUNT
        for coefficient, item in zip(coefficients, self.pending):
            for index, value in enumerate(item["delta"]):
                change[index] += coefficient * value
        self.model = {
            "schema_version": 1,
            "model_type": "onnx_probability_calibration",
            "base_model_contract": MODEL_CONTRACT,
            "version": int(self.model.get("version", 0)) + 1,
            "parameters": [old + update for old, update in zip(self.model["parameters"], change)],
            "last_aggregation": {"clients": len(self.pending), "weights": coefficients},
        }
        self.pending = []
        self._write(self.model_path, self.model)
        self._write(self.pending_path, self.pending)
        return self.model


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, value):
            encoded = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorised(self):
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if supplied and hmac.compare_digest(supplied, state.token):
                return True
            self._send(401, {"error": "valid research token required"})
            return False

        def do_GET(self):
            if not self._authorised():
                return
            if self.path == "/health":
                return self._send(200, {"status": "ok", "pending_clients": len(state.pending)})
            if self.path == "/model/calibration":
                return self._send(200, state.model)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorised():
                return
            try:
                if self.path != "/updates":
                    return self._send(404, {"error": "not found"})
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 250_000:
                    raise ValueError("invalid payload size")
                self._send(202, state.accept(json.loads(self.rfile.read(size))))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, template, *args):
            print(f"{self.client_address[0]} - {template % args}")
    return Handler


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default="outputs/optional_learning_server")
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--token-file", required=True, help="UTF-8 file containing the shared local research token")
    return parser.parse_args()


def main(config):
    token = Path(config.token_file).read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise ValueError("research token must contain at least 24 characters")
    state = LearningState(config.state_dir, config.min_clients, token)
    server = ThreadingHTTPServer(("127.0.0.1", config.port), make_handler(state))
    print(f"PocketIQ optional-learning server: http://127.0.0.1:{config.port}")
    server.serve_forever()


if __name__ == "__main__":
    main(arguments())
