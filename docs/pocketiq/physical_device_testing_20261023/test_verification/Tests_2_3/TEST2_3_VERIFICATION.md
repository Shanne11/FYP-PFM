# Test 2/3 verification — ACTM Review, Smart Note and correction audit

Source video: Private recording retained outside this public repository
SHA-256: `E6D26B67D20AA89751822E641E6C9A37B73A0E94E8E3CC525DDD3C860B236591`  
Duration: 118.1 seconds  
Resolution: 292 × 642

## Synthetic test case

| Field | Observed value |
|---|---|
| Merchant | `ZXQ TEST UNKNOWN 8472` |
| Amount | RM7.89 spending transaction |
| Account | Cash Wallet |
| Initial displayed category | Other Spending |
| Smart Note | `Bus fare for physical-device verification` |
| Note-enhanced recommendation | Shopping |
| Displayed recommendation confidence | 86% |
| User-selected final category | Transportation |

## Observed sequence

1. A synthetic unknown-merchant transaction is created with AI auto-detection.
2. PocketIQ opens the Smart Note clarification workflow and marks the transaction as needing a note.
3. Review Queue contains one transaction requiring quick review.
4. The user enters the synthetic bus-fare purpose note.
5. The AI Recommendation screen recommends `Shopping` at 86% confidence.
6. The user does not accept the incorrect recommendation and selects `Transportation`.
7. PocketIQ displays `Correction Saved`.
8. The transaction's **Why this? / Category Explanation → Timeline** view shows:
   - Initial prediction: Other Spending
   - ACTM clarification requested
   - Smart Note added
   - User correction saved
   - Current final category: Transportation

## Verification results

| Check | Result |
|---|---|
| Synthetic transaction created | PASS |
| Transaction entered Review Queue | PASS |
| ACTM clarification reason displayed | PASS |
| Smart Note accepted | PASS |
| Frozen-model recommendation displayed | PASS |
| User able to reject incorrect recommendation | PASS |
| Correct category selected | PASS |
| Correction saved | PASS |
| Audit Timeline contains ACTM, note and correction events | PASS |
| Audit Timeline reconfirmed after leaving and relaunching the app | PASS — `AdditionalTest2_3.mp4` |
| Android Settings Force stop visibly performed before reconfirmation | NOT SHOWN |

Overall Test 2 status: **PASS**.  
Overall Test 3 status: **PASS for application-relaunch persistence**. `AdditionalTest2_3.mp4` reconfirms the Smart Note, saved correction and final category after PocketIQ is left and relaunched. The remaining terminology limitation is that the recording does not visibly perform Android Settings **Force stop**, so it must not be cited as a settings-level forced-stop test.

## Research interpretation boundary

This single synthetic case is functional verification, not an accuracy measurement. It must not be reported as evidence that ACTM or Smart Notes generally improve categorisation.

The observed result is mixed and should be preserved honestly:

- ACTM successfully routed an unknown transaction for review.
- The Smart Note path produced a confident but incorrect `Shopping` recommendation.
- Human correction changed the final category to the contextually appropriate `Transportation` category.
- The audit trail recorded the sequence.

This supports the need for user confirmation and auditability; it does not demonstrate universal Smart Note reliability.

## Persistence follow-up completed

`AdditionalTest2_3.mp4` shows that `Smart Note added`, `User correction saved` and `Current final category: Transportation` remain available after leaving and relaunching PocketIQ. A separate settings-level Force stop recording is optional only if the thesis specifically claims that exact operating-system action was tested.

