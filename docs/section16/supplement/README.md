# Section 16 Q1–Q20 — Supporting audit supplement (12 October 2026)

This supplement includes six unchanged audit outputs found in
`Section16_Evidence_Package_PROVISIONAL_2026-10-11.zip` but not separately bundled
inside `Section16_Evidence_Package_FINAL_v2_2026-10-12.zip`.

The original audit records are retained byte-for-byte to preserve provenance.
They are supporting calculation artifacts, NOT the obsolete provisional status
narrative. For current Q6, Q12, Q15 and repository statuses use ONLY the FINAL v2
manifest/evidence matrix/audit. The final research/source commits remain:
Python e4aabb55a7d779e2e09dcac59c2a5746895c2149;
Flutter 11daf49f491e70f9f7057cc3cf3bc3616990bb96.

Confirmed consistency checks:
- Q5 historical canonical distribution: train total 7,494; max 1,623; min 184.
- Q8/Q9 selector recheck: 5 strategies x 3 seeds, each with 693 selections.
- Q13 participation: 30 seed-round rows, 150 participating clients per method;
  same client IDs and sample counts in every row.
- Q16 weights: four quartile rows; smallest quartile shows a small mean
  negative relative adjustment, but this is NOT a client-accuracy harm test.
- Q19 model contract comparison: 105 research features and 4,096 deployment features.

This supplement does not include the standalone PocketIQ ONNX binary and
cannot independently establish merchant-disjointness, usability, or formal
privacy protection. Use it alongside the FINAL v2 evidence package, not instead.

Archive file names match original provisional package member paths. Hashes
below are SHA-256 of each original audit file. See manifest.json.
