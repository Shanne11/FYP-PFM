# Section 16 Evidence Package — Provisional

Purpose: link every Section 16 numerical result and implementation claim to an experiment output, source implementation or test record.

Status: provisional because the Python and Flutter working trees contain uncommitted changes. The 26 declared artefacts match the existing SHA-256 manifest, but a final freeze must be regenerated after both repositories are committed.

Critical corrections:
- Q6: the original matrix/report incorrectly stated zero ID overlap; split comparisons show 106, 166 and 40 repeated ID values, zero exact available-record overlaps, and no demonstrated merchant-disjoint splitting.
- Q12: the supplied final-wrong formula reproduces 874 cases; the stricter implemented wrong-before-and-after metric reproduces 845. Neither supports 900.
- Q15: matched final-test client accuracy reproduces 0 improved, 0 declined and 450 unchanged client-seed cases, not 15/19/416.

Evidence boundaries:
- BudgetWise notes are retrospective source-note proxies, not participant clarification responses.
- PocketIQ Flutter uses a separate 4,096-feature, 14-category ONNX contract.
- The Tier A demonstration trace is not a PocketIQ on-device audit record.
- Q18 passes source inspection and a direct programmatic sync rejection test, but the Flutter changes still require a final commit.

Open `audit/Section16_Evidence_Matrix.xlsx` for the main review view.

12 October 2026 Q6 revision: The matrix CSV and XLSX, audit Markdown, and bundled audit-generator script are corrected in this *provisional copy*. The user's local repositories are not changed by this ZIP. The 26 declared research artifacts, their SHA-256 hashes and the original split-overlap CSV remain unchanged. Consult the external corrected-ZIP .sha256.txt file for archive integrity; the stale internal checksum from the original ZIP was omitted.
