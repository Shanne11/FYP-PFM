# Section 16 Evidence and Question Traceability

This directory is the browsable GitHub index for the final Section 16 evidence. It links each supervisor question to the verified report, matrix, script, test record, or audit output that supports the answer. The historical intended release identifier is `section16-evidence-final-v2-2026-10-12`. On 22 October 2026, the GitHub releases endpoint returned no releases and that tag could not be found. The source index remains available here, but the full archive is not verified as downloadable from GitHub.

## Provenance

The research results were produced from these source commits:

- Tier A Python research source: `e4aabb55a7d779e2e09dcac59c2a5746895c2149`
- PocketIQ Flutter source: `11daf49f491e70f9f7057cc3cf3bc3616990bb96`

The commit containing this directory is a later documentation commit. It does not replace either research-source commit and does not indicate that the experiments were rerun.

## Final package

- Release archive: `Section16_Evidence_Package_FINAL_v2_2026-10-12.zip`
- Archive SHA-256: `af57c804d59404c949f4875b3ca61900eda8fe8953a96185417a1eb38764aeca`
- Manifest validation: 71 of 71 package artifacts matched before release assembly.
- Declared research evidence: 26 of 26 artifacts matched their recorded SHA-256 hashes.
- Original transaction-level CSV verification: 28 of 28 manifest entries matched.

The release also contains the Q1-Q20 traceability supplement and the thesis-analysis document. The supplement preserves six historical audit outputs byte-for-byte; it does not restore obsolete provisional statuses.

## Current verification status

The final Q1-Q20 matrix contains eight `Verified` questions and twelve `Verified with limitation` questions. It contains no `Needs correction` question.

| Question | Final status | Verified result |
| --- | --- | --- |
| Q6 | Verified with limitation | Repeated transaction IDs are 106, 166, and 40 across the three partition comparisons. Exact available-record overlap is zero in every comparison. Merchant-disjoint splitting is not established. |
| Q12 | Verified with limitation | Of 1,736 retrospective note-used transaction-seed evaluations, 874 had increased confidence with a wrong final prediction. The stricter wrong-before-and-after count is 845. |
| Q15 | Verified with limitation | The matched client-level comparison contains 0 improved, 0 declined, and 450 unchanged client-seed accuracies. Global accuracy and Macro F1 do not improve. |
| Q18 | Verified with limitation | Remote optional-learning upload is disabled and direct `syncPending()` calls are rejected. Four focused Flutter tests passed; the complete Flutter suite and unrelated network features were not tested. |

## Repository contents

- `audit/section16_evidence_matrix.csv` is the machine-readable Q1-Q20 evidence map.
- `audit/Section16_Evidence_Matrix.xlsx` is the formatted evidence matrix.
- `audit/Section16_Evidence_Audit.md` explains the verification findings and limitations.
- `manifest/final_evidence_manifest.json` records package hashes and source provenance.
- `manifest/source_repositories.json` records the research-source repository states used during package assembly.
- `manifest/test_results.json` records the targeted Python, focused Flutter, and transaction-level CSV checks.
- `verification/Section16_Independent_Verification.py` is the independent verification script.
- `verification/independent_csv_verification.txt` and `verification/Section16_Verification_console_output.txt` preserve the corresponding outputs.
- `supplement/` contains the six cross-referenced historical audit outputs and their manifest.
- `thesis/` contains the question-by-question thesis analysis aligned with the final matrix.

Paths recorded inside the final evidence matrix are package-relative. Some large or binary evidence files are therefore present in the release archive rather than duplicated in this directory. Source files and ordinary experiment outputs remain at their original repository paths.

## Evidence boundary

Tier A Smart Notes are retrospective source-note proxies. They are not participant responses entered after an ACTM prompt and do not establish user acceptance or usability. PocketIQ uses a separate frozen 4,096-feature, 14-category ONNX deployment contract; the 105-feature, 13-category Tier A research checkpoint is not deployed in the application.

## Test limitation

Eight targeted Python verification tests and four focused Flutter tests passed. The completed experiments were not rerun during documentation reconciliation. The full Flutter test suite was not run, and the privacy check did not test unrelated feedback, sharing, export, or other network features.
