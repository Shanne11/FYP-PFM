# Test 1 verification — cold-restart persistence

Source video: Private recording retained outside this public repository
SHA-256: `756037D3BE04399FF01BA18BD44FC1F5335A95DB13CF775047CE3306812841A7`  
Duration: 33 seconds  
Resolution: 386 × 848

## Observed sequence

1. The recording begins on the `Physical Test Coffee` detail page and shows the previously saved optional transaction description/note text.
2. Android's PocketIQ application-details page is opened.
3. The **强行停止** (Force stop) control becomes disabled, which is consistent with the application process having been force-stopped.
4. PocketIQ is opened again.
5. The Transactions page displays the same `Physical Test Coffee` transaction, RM12.34, dated 23 October, under Cash Wallet.

## Verdict

| Check | Result |
|---|---|
| Real physical device used | PASS |
| PocketIQ force-stopped | PASS |
| PocketIQ reopened | PASS |
| Transaction persisted after force stop | PASS |
| Optional transaction description/note visible before force stop | PASS |
| Optional transaction description/note visibly reconfirmed after restart | PASS — `UpdatedTest1.mp4` |

Overall status: **PASS**, after review of `UpdatedTest1.mp4`.

Together, `Test1.mp4` and `UpdatedTest1.mp4` prove cold-restart persistence of the transaction record and its optional description/note field.

This field is displayed under **Description** and must not be relabelled as an ACTM Smart Note. ACTM Smart Note persistence is assessed separately using the `ZXQ TEST UNKNOWN 8472` clarification record.

