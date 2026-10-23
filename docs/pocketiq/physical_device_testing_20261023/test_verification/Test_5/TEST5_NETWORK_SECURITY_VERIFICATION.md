# Test 5 — PocketIQ network and security verification

Date: 23 October 2026  
Result: **PASS WITH LIMITATIONS**

This is an engineering privacy/security audit, not a professional penetration test.

## Results

| Check | Result | Evidence or limitation |
|---|---|---|
| Optional-learning remote upload flag | PASS | `OptionalLearningService.remoteSyncEnabled == false` |
| Direct `syncPending()` rejection | PASS | Focused Flutter tests passed 2/2; direct call throws `StateError` |
| Local categorisation and Smart Note processing | PASS | Focused offline-capable regression tests passed 24/24 |
| Complete Flutter regression suite | PASS | 77/77 tests passed |
| Ordinary ADB-shell access to app-private storage | PASS | Both `ls` and `stat` returned `Permission denied` |
| Exported Android components | PASS WITH LIMITATION | Launcher activity is intentionally exported; dependency components are non-exported or permission-protected |
| First-party endpoint/client inventory | PASS | No hard-coded production API endpoint or first-party Dart HTTP/socket client found |
| Embedded-secret pattern scan | PASS | No API-key, client-secret, Authorization, bearer-token or `sk-` pattern found in first-party Dart, Kotlin or main-manifest source |
| Controlled runtime network observation | PASS WITH LIMITATION | PocketIQ UID counters changed by 0 bytes and 0 packets during a 12-second force-stop/relaunch observation |
| Installed APK permission minimisation | NEEDS HARDENING | Profile APK declares network permissions and legacy storage/phone-state permissions; dangerous legacy permissions were runtime-denied |
| Negative unauthenticated backend access | NOT APPLICABLE | PocketIQ has no configured controlled backend endpoint to test |
| Professional penetration assessment | NOT CONDUCTED | Outside this engineering test scope |

## Optional-learning privacy result

The focused test file `test/optional_learning_upload_disabled_test.dart` verified two conditions:

1. `remoteSyncEnabled` is `false`.
2. A direct programmatic call to `syncPending()` is rejected with an upload-disabled `StateError`.

The implementation performs local calibration without serialising an upload payload or creating a pending synchronisation record while remote synchronisation is disabled.

## Static Android security result

The merged installed profile APK exposes only the expected launcher activity without a special permission. Share, URL-launcher, ML Kit and initialisation components are non-exported. AndroidX's exported profile-installer receiver is protected by the privileged `android.permission.DUMP` permission.

The permission audit found:

- `INTERNET` and `ACCESS_NETWORK_STATE` are declared and install-time granted;
- biometric permissions are declared for the app-lock function;
- `READ_EXTERNAL_STORAGE`, `WRITE_EXTERNAL_STORAGE` and `READ_PHONE_STATE` are declared but runtime-denied on the tested Android 16 device;
- Android ML Kit/DataTransport dependencies contribute network-capable components.

Therefore, PocketIQ must not be described as technically unable to access the network. The defensible claim is narrower: **the optional-learning upload pathway is disabled and no PocketIQ UID traffic was observed during the bounded launch observation**.

The legacy storage and phone-state declarations should be removed from a future release manifest if dependency compatibility testing confirms that they are unnecessary. This audit did not modify the manifest because Test 5 was a verification task.

## Runtime network observation

Android UID `10501` counters were captured after force-stopping PocketIQ and again after launching it and waiting 12 seconds:

| Counter | Before | After | Difference |
|---|---:|---:|---:|
| Received bytes | 88,209 | 88,209 | 0 |
| Transmitted bytes | 4,320,618 | 4,320,618 | 0 |
| Received packets | 1,474 | 1,474 | 0 |
| Transmitted packets | 2,769 | 2,769 | 0 |

This is a bounded observation, not full packet capture. It cannot prove that every application feature remains network-silent over unlimited time or under every dependency behaviour.

## Private-storage protection

Ordinary ADB-shell attempts to list or stat `/data/user/0/com.pocketiq.pocketiq` were rejected with `Permission denied`. The debug/profile-only `run-as` facility was not counted as an unauthorised bypass because it depends on the deliberately debuggable engineering build.

## Negative endpoint boundary

No controlled PocketIQ server endpoint is configured. Consequently, there is no legitimate target for an unauthenticated remote-access rejection test. No third-party or production system was probed. Direct rejection of `syncPending()` is verified separately, but it is not presented as a server-side penetration test.

## Final Test 5 verdict

Test 5 is complete as an engineering audit and is **PASS WITH LIMITATIONS**:

- optional-learning upload-disabled behaviour passed;
- local categorisation tests passed;
- app-private storage resisted ordinary ADB-shell access;
- exported components did not reveal an unprotected sensitive component;
- no PocketIQ UID traffic was observed during the bounded launch test;
- APK permission minimisation remains a hardening item;
- no backend penetration or professional security assessment was performed.

Detailed command outcomes are recorded in `TEST5_TECHNICAL_LOG.txt`.

