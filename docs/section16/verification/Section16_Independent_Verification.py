"""Reproduce Section 16 Q8-Q12 and Q15 verification from uploaded original CSV ZIP.

Usage: python Section16_Independent_Verification.py Original_Prediction_ACTM_CSVs_Seeds_42_52_62.zip
Requires: pandas (and Python standard library).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import pandas as pd


def main(zip_path: Path):
    with tempfile.TemporaryDirectory(prefix="section16_verify_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            assert bad is None, f"Corrupt member: {bad}"
            for member in z.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
            z.extractall(root)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for artifact in manifest["files"]:
            path = root / artifact["packaged_path"]
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], path
            assert path.stat().st_size == artifact["bytes"], path
            if path.suffix.lower() == ".csv" and "data_rows" in artifact:
                assert len(pd.read_csv(path)) == artifact["data_rows"], path
        print(f"Manifest validation: {len(manifest['files'])}/{len(manifest['files'])} passed")
        base = root / "outputs" / "actm_selection_baselines"
        df = pd.read_csv(base / "actm_selection_transaction_outcomes.csv")
        runs = pd.read_csv(base / "actm_selection_runs.csv")
        assert len(df) == 34620
        assert not df.duplicated(["seed", "strategy", "source_index"]).any()
        assert sorted(df.seed.unique().tolist()) == [42, 52, 62]
        assert len(runs) == 15
        summary = []
        for (seed, strategy), group in df.groupby(["seed", "strategy"]):
            assert len(group) == 2308
            sel = group.loc[group.prompted]
            used = group.loc[group.note_used]
            assert len(sel) == 693
            assert (used.prompted & used.note_available).all()
            wc = (~used.correct_before) & used.correct_after
            cw = used.correct_before & (~used.correct_after)
            ww = (~used.correct_before) & (~used.correct_after)
            cc = used.correct_before & used.correct_after
            confidence_increased = used.confidence_after > used.confidence_before
            calculated = {
                "prompts": len(sel),
                "notes_used": len(used),
                "prompt_precision": (~sel.correct_before).mean(),
                "note_acceptance_rate": len(used)/len(sel),
                "wrong_to_correct": int(wc.sum()),
                "correct_to_wrong": int(cw.sum()),
                "wrong_to_different_wrong": int((ww & (used.predicted_before != used.predicted_after)).sum()),
                "wrong_unchanged": int((ww & (used.predicted_before == used.predicted_after)).sum()),
                "correct_unchanged": int(cc.sum()),
                "net_corrections": int(wc.sum()-cw.sum()),
                "correction_yield_per_100_prompts": 100*wc.sum()/len(sel),
                "overall_accuracy": group.correct_after.mean(),
            }
            saved = runs.loc[(runs.seed == seed) & (runs.strategy == strategy)].iloc[0]
            for key, value in calculated.items():
                assert abs(saved[key] - value) < 1e-9, (seed, strategy, key, saved[key], value)
            summary.append({"seed": seed, "strategy": strategy, **calculated,
                            "more_confident_final_wrong": int((confidence_increased & (~used.correct_after)).sum()),
                            "more_confident_wrong_before_after": int((confidence_increased & ww).sum())})
        result = pd.DataFrame(summary)
        means = result.groupby("strategy")[["prompt_precision", "correction_yield_per_100_prompts", "net_corrections", "overall_accuracy"]].mean()
        print("\nQ8-Q9 three-seed means (recalculated from transaction rows):")
        print(means.round(6).to_string())
        proposed = result.loc[result.strategy == "Proposed ACTM"]
        totals = proposed[["notes_used", "wrong_to_correct", "correct_to_wrong", "wrong_to_different_wrong", "wrong_unchanged", "correct_unchanged", "more_confident_final_wrong", "more_confident_wrong_before_after"]].sum().astype(int)
        print("\nQ11-Q12 Proposed ACTM totals:")
        print(totals.to_string())
        assert totals["notes_used"] == 1736
        assert totals["wrong_to_correct"] == 56
        assert totals["correct_to_wrong"] == 54
        assert totals["wrong_to_different_wrong"] == 205
        assert totals["wrong_unchanged"] == 1214
        assert totals["correct_unchanged"] == 207
        assert totals["more_confident_final_wrong"] == 874
        assert totals["more_confident_wrong_before_after"] == 845
        print("\nQ15 full versus without_heuristic_adjustment:")
        for seed in [42, 52, 62]:
            sub = root / "outputs" / "ablations" / f"seed_{seed}"
            left = pd.read_csv(sub / "full" / "predictions.csv")
            right = pd.read_csv(sub / "without_heuristic_adjustment" / "predictions.csv")
            assert len(left) == len(right) == 2308
            assert left.transaction_id.equals(right.transaction_id)
            assert left.actual.equals(right.actual)
            different_predictions = int((left.predicted != right.predicted).sum())
            different_confidence = int((left.confidence != right.confidence).sum())
            assert different_predictions == 0
            print(f"  seed {seed}: {different_predictions} category changes; {different_confidence} confidence values differ")
        print("\nPASS: all original CSV comparisons are consistent with corrected Section 16 Q8-Q12 and Q15.")
        print("Note: per-client counts require the separately supplied client-level audit/client mapping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    main(args.zip)
