# PocketIQ physical-device evidence report

Date: 23 October 2026  
Evidence status: physical-device engineering verification with stated limitations

## Tested build and device

| Item | Recorded value |
|---|---|
| Source repository | `[private local path omitted]` |
| Git branch | `main` |
| Source commit | `11daf49f491e70f9f7057cc3cf3bc3616990bb96` |
| Build mode | Flutter profile |
| Application ID | `com.pocketiq.pocketiq` |
| App version | 1.0.0 (version code 1) |
| Device | OPPO CPH2689 |
| Android | 16 (API 36) |
| Flutter | 3.44.4 stable |

The device serial is deliberately omitted from this report.

## Video evidence

Source recording: Private recording retained outside this public repository

| Property | Value |
|---|---:|
| Duration | 63.7 seconds |
| Resolution | 386 × 848 |
| Reported frame rate | 39.89 fps |
| SHA-256 | `9B147A87E174725D92E09440F570B7033822327F960AF0C829A08367F25A176B` |

The recording uses the synthetic transaction `Physical Test Coffee`, amount RM12.34. Representative local review frames were inspected at 10, 30, 45, 55 and 62 seconds.

### Demonstrated by the recording

- Physical Android application launch and interaction.
- Manual creation of a spending transaction with amount RM12.34.
- Saving and listing the transaction.
- Rule-based categorisation shown in the transaction list.
- Opening/editing the saved transaction.
- Entry of the note `Coffee and breakfast during device testing`.
- Saving the note and displaying it on the transaction detail page.
- Display of PocketIQ categorisation information on the detail page.

The UI presents `Food & Dining` as the broader category and `Beverages & Snacks` in the transaction/category presentation. This report does not reinterpret those UI labels as two conflicting research predictions.

### Not demonstrated by the recording

- Cold restart followed by confirmation that the record remains available.
- ACTM Review workflow.
- User correction plus audit-history persistence.
- CSV export/download.
- Same-model before/after probability comparison.
- Optional-learning direct-call rejection.
- Network traffic or unauthenticated-access behaviour.

The recording is supporting evidence, not proof of every PocketIQ requirement.

## Persistence evidence

Android application storage metadata confirms the presence of:

- `app_flutter/pocketiq.sqlite`
- `app_flutter/pocketiq_transactions_1791986550401.csv`
- Flutter secure-storage preference files

No database, CSV or secure-storage contents were read. The video demonstrates save-and-display persistence within the recorded application session. Cold-start persistence remains to be tested separately.

## Memory evidence

| Point | Total PSS | Total RSS | Swap PSS |
|---|---:|---:|---:|
| Initial profile capture | 332,898 KB | 437,396 KB | 40,568 KB |
| Post-workflow capture | 185,219 KB | 272,964 KB | 55,933 KB |

The post-workflow PSS is approximately 180.9 MiB. The difference must not be interpreted as memory saved by the workflow: the captures occurred at different warm-up/garbage-collection states and were not a repeated controlled benchmark.

## Rendering evidence

The initial capture contained only one frame and is not a meaningful performance sample. The post-workflow Android rendering statistics recorded:

| Metric | Result |
|---|---:|
| Frames rendered | 117 |
| Janky frames | 5 (4.27%) |
| 50th percentile | 5 ms |
| 90th percentile | 11 ms |
| 95th percentile | 15 ms |
| 99th percentile | 57 ms |

These are Android frame-rendering diagnostics for one short profile-mode workflow. They are not end-to-end inference-latency measurements.

## Runtime warnings and errors

No Flutter exception, application crash or application error was present in the captured post-workflow warning/error log. Three Android view/surface resize warnings were recorded. They did not terminate the application during the observed workflow.

The build also emitted a future-compatibility warning: the ONNX Runtime and ML Kit plugins currently apply the Kotlin Gradle Plugin directly and will eventually need migration to Kotlin's built-in Gradle plugin. This was not a build failure.

## Battery evidence

| Point | Charge level | Temperature | Status |
|---|---:|---:|---|
| Baseline | 76% | 34.8 °C | Charging |
| Post-workflow | 76% | 35.2 °C | Charging |

No defensible battery-consumption result can be reported because the device was charging and the observation lasted about one minute. A longer controlled, unplugged, repeated test is still required.

## Defensible status update

| Report item | Updated status |
|---|---|
| Physical-device partial end-to-end workflow | Verified with limitation |
| Transaction creation and save | Verified |
| Note entry and same-session display | Verified |
| Cold-start transaction persistence | Verified by `Test1.mp4` |
| Cold-start optional transaction description/note persistence | Verified by `UpdatedTest1.mp4` |
| Post-relaunch ACTM Smart Note/audit persistence | Verified by `AdditionalTest2_3.mp4` |
| ACTM Review and Smart Note workflow | Verified by `Test2_3.mp4` |
| Correction and same-session audit Timeline | Verified by `Test2_3.mp4` |
| Correction/audit Timeline after app relaunch | Verified by `AdditionalTest2_3.mp4`; Android Settings Force stop was not visibly recorded |
| Private CSV creation on physical device | Verified by `Test4.mp4` |
| User-accessible CSV share | Verified by `UpdatedTest4.mp4`: Android share interface, WhatsApp receipt and successful spreadsheet opening |
| Direct local-device CSV save | Verified with limitation: 77/77 tests passed, APK installed and user confirmed successful local download; no separate video of the Downloads action |
| Rendering responsiveness | Measured once; exploratory |
| Memory | Measured once; exploratory |
| Battery consumption | Not assessed defensibly |
| Network/security engineering audit | Pass with limitations; see `TEST5_NETWORK_SECURITY_VERIFICATION.md` |

## Required follow-up

1. Run an unplugged repeated battery protocol of predetermined duration.
2. Harden and re-audit the release manifest to remove unnecessary legacy storage/phone-state permissions, and assess whether dependency-added network permissions can be removed safely.
3. Only if a settings-level forced-stop claim is required, record Android Settings → Apps → PocketIQ → Force stop before reopening the persisted Timeline. The current evidence verifies application-relaunch persistence, not that exact operating-system action.

