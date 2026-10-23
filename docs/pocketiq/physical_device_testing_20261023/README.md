# PocketIQ Physical-Device Verification — Tests 1–5

Evidence date: **23 October 2026**. This is a **public-safe textual subset** of the evidence supporting **Chapter 5 §5.8**, not the thesis report or a replacement for the private original archive.

| Test | Result | Location |
|---|---|---|
| Test 1 — transaction persistence | PASS | `test_verification/Test_1/` |
| Test 2 — ACTM Review and Smart Note | PASS | `test_verification/Tests_2_3/` |
| Test 3 — correction/audit persistence | PASS WITH LIMITATION | `test_verification/Tests_2_3/` |
| Test 4 — CSV share and local save | PASS | `test_verification/Test_4/` |
| Test 5 — network/security engineering audit | PASS WITH LIMITATIONS | `test_verification/Test_5/` |

**Important limitations:** Test 3 shows persistence after application relaunch but not a visibly recorded Android Settings Force stop for the audit test. Test 4's Downloads save is user-confirmed, not separately video-recorded. Test 5 is an engineering review, **not a professional penetration test**; APK permission hardening remains. The 77/77 Flutter regression result applies to the later tested local build and does not prove that unpushed CSV-save changes are present in a remote source repository.

**Publication rules:** All original video recordings, private review screenshots, exported CSV transaction contents, nested ZIP archives, raw system diagnostics, device identifiers and local-only HTML review pages are excluded. The verification notes below have local file paths and app-private storage paths removed. Reports can refer to private source-video files and their reported hashes; the videos themselves are not included.

**Research distinction:** PocketIQ uses a separate 14-category, 4,096-feature frozen ONNX deployment model. Its functional tests are not evidence of Tier A 13-category federated method accuracy or of Expanded V2 genuine Smart Note performance. Do not merge those study results.

Read `TEST1_TO_5_COMPLETION_SUMMARY.md` and then the corresponding verification report. For audit-level access to video and full diagnostics, use the privately held source package with supervisor authorisation.
