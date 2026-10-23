# PocketIQ local-device CSV save implementation

Date: 23 October 2026  
Repository: `[private local path omitted]`  
Branch: `main`

## Change

PocketIQ now offers two distinct actions on **Settings → Data export**:

1. **Save CSV to device** — opens Android's system document picker (`ACTION_CREATE_DOCUMENT`) so the user can choose a local folder such as Downloads and confirm the filename.
2. **Share CSV** — retains the previously verified Android share workflow for Drive, WhatsApp, email and other trusted applications.

The local-save implementation uses Android's Storage Access Framework. It does not request broad storage permission and does not display PocketIQ's private `/data/user/0/...` path.

Changed source:

- `lib/features/settings/view/settings_control_pages.dart`
- `android/app/src/main/kotlin/com/pocketiq/pocketiq/MainActivity.kt`
- `test/data_export_share_test.dart`

No categorisation, Smart Note, ACTM, database, optional-learning or research-result logic was changed.

## Verification

| Check | Result |
|---|---|
| Dart formatting | PASS |
| Focused CSV save/share tests | PASS (3/3) |
| Static analysis of changed Dart source/test | PASS — no issues found |
| Full Flutter regression suite | PASS (77/77) |
| Git whitespace check | PASS |
| Profile APK native compilation | PASS |
| Installation over existing app using `adb install -r` | PASS |
| Application launch | PASS |
| Physical selection of Downloads and successful local download | PASS — user confirmed on the installed physical-device build |

Profile APK:

- Path: `[private local path omitted]`
- Size: 231,909,392 bytes
- SHA-256: `061BC4FCBD442EA38FEC0EC6295A711B7A6075CC0441B196AB0CC683F99B772D`

The updated APK was installed without requesting data removal. The user subsequently confirmed that saving to the local device completed successfully. This confirmation was not captured in a separate video, so it is recorded as user-confirmed functional evidence rather than recorded visual evidence. The implementation changes remain uncommitted and unpushed.

## Manual confirmation steps

1. Open PocketIQ **Settings → Data export**.
2. Tap **Save CSV to device**.
3. In Android's file picker, select **Downloads**.
4. Keep or edit the suggested `pocketiq_transactions_<timestamp>.csv` filename.
5. Tap **Save**.
6. Open the Files app → Downloads and confirm the CSV opens.

Use synthetic data for recordings and keep CSV files containing financial information private.

