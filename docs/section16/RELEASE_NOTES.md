# Section 16 Final Evidence Package v2

This release preserves the final reconciled evidence for the twenty supervisor questions. It is a documentation and evidence release. The completed experiments were not rerun and the verified research results were not modified.

## Research-source provenance

- Tier A Python source: `e4aabb55a7d779e2e09dcac59c2a5746895c2149`
- PocketIQ Flutter source: `11daf49f491e70f9f7057cc3cf3bc3616990bb96`

The Git tag identifies the later documentation commit containing the browsable evidence index. It must not be interpreted as the commit that produced the research results.

## Verified corrections

- Q6: repeated transaction-ID values are 106, 166, and 40; exact available-record overlaps are zero; merchant-disjoint splitting is not established.
- Q12: 874 of 1,736 retrospective note-used evaluations had increased confidence with a wrong final prediction; the stricter wrong-before-and-after count is 845.
- Q15: 0 client-seed cases improved, 0 declined, and 450 were unchanged; final held-out categories, global accuracy, and Macro F1 do not improve in the matched comparison.
- Q18: optional-learning remote upload is disabled and direct `syncPending()` calls are rejected.

## Verification summary

- 71 of 71 package artifacts matched the final package manifest.
- 26 of 26 declared research artifacts matched their SHA-256 values.
- 28 of 28 original transaction-level CSV manifest entries matched.
- Eight targeted Python tests passed.
- Four focused Flutter tests passed in an isolated tracked-source copy.

## Limitations

Tier A Smart Notes are retrospective source-note proxies, not participant responses entered after an ACTM prompt. The complete Flutter test suite was not run. The Flutter privacy check covered the optional-learning and local-calibration pathway only; unrelated network, feedback, sharing, and export features were outside its scope.

## Assets

- `Section16_Evidence_Package_FINAL_v2_2026-10-12.zip` contains the complete frozen package.
- `Section16_Evidence_Package_FINAL_v2_2026-10-12.zip.sha256.txt` contains its standalone checksum.
- `Section16_Q1-Q20_Traceability_Supplement_2026-10-12.zip` contains six additional historical audit outputs preserved byte-for-byte.
- `ShannellWongSiYuan_THESIS_ANALYSIS_With_Question_Evidence_FINAL.docx` contains the question-by-question thesis analysis aligned to the final evidence matrix.
- `RELEASE_ASSET_SHA256.txt` records checksums for the principal release assets.
