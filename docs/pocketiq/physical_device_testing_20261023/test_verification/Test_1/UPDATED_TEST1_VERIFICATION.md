# Updated Test 1 verification — cold-restart persistence

Source video: Private recording retained outside this public repository
SHA-256: `0F8833438623BDA16524F405A57B179855A2DE50A718202CD2032D57282AE170`  
Duration: 18.4 seconds  
Resolution: 386 × 848

## Observed sequence

1. PocketIQ is opened from the physical phone's launcher following the force-stop/restart sequence recorded in `Test1.mp4`.
2. The Transactions list contains both persisted synthetic records.
3. `Physical Test Coffee`, RM12.34, is opened.
4. The detail page displays `Coffee and breakfast during device testi...` under **Description**.
5. The Explanation/Timeline view remains accessible and shows the initial prediction, transaction edit and current final category.

## Verification results

| Check | Result |
|---|---|
| PocketIQ reopened on physical device | PASS |
| Synthetic transaction persisted | PASS |
| RM12.34 amount persisted | PASS |
| Category persisted | PASS |
| Optional transaction description/note text persisted | PASS |
| Explanation/Timeline accessible after restart | PASS |

Overall Test 1 status: **PASS**.

## Terminology boundary

The persisted text is displayed by PocketIQ under **Description**, and its Timeline event is `Transaction edited`. It must therefore be described as the transaction's optional description/note field, not as an ACTM Smart Note clarification record.

The ACTM Smart Note record is demonstrated separately by `Test2_3.mp4` for `ZXQ TEST UNKNOWN 8472`. Its cold-restart audit persistence remains a separate check until the post-restart Timeline shows `Smart Note added` and `User correction saved`.

