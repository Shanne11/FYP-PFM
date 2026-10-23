# Additional Test 2/3 verification — post-relaunch audit persistence

Source video: Private recording retained outside this public repository
SHA-256: `25B7CBEBD8C96EB438617F580FF273A5D2E631700D17E6D0BF99CB35BB740514`  
Duration: 18.3 seconds  
Resolution: 386 × 848

## Observed evidence

The recording uses the same synthetic transaction as `Test2_3.mp4`:

- Merchant: `ZXQ TEST UNKNOWN 8472`
- Amount: RM7.89 spending transaction
- Persisted final category: `Transportation`
- Persisted Smart Note: `Bus fare for physical-device verification`

The recording first shows the saved transaction in the transaction list with the final `Transportation` category. It then opens the category explanation, leaves PocketIQ through Android's recent-apps interface, returns to the application, reopens the transaction, and displays the audit Timeline again.

The post-relaunch Timeline contains:

1. Initial prediction: `Other Spending`
2. ACTM clarification requested because the merchant was not recognised by local history or rules
3. Smart Note added: `Bus fare for physical-device verification`
4. User correction saved after the AI suggested `Shopping` at 86% confidence
5. Current final category: `Transportation`

## Verification result

| Check | Result |
|---|---|
| Corrected transaction remains in the transaction list | PASS |
| Final `Transportation` category remains saved | PASS |
| Smart Note remains present in the Timeline | PASS |
| ACTM clarification event remains present | PASS |
| User-correction event remains present | PASS |
| Complete Timeline can be reopened after leaving and relaunching PocketIQ | PASS |
| Android Settings **Force stop** action visibly recorded | NOT SHOWN |

Overall result: **PASS for application-relaunch persistence of the ACTM Smart Note, user correction and audit Timeline**.

The video shows Android recent-apps/app relaunch behaviour, not the Android Settings **Force stop** control. It therefore supports persisted application state after relaunch, but should not be described as direct evidence of a settings-level forced-stop test.

## Research interpretation boundary

This is one synthetic functional-verification case. It demonstrates persistence and auditability; it does not measure classification accuracy, Smart Note effectiveness, ACTM effectiveness or real-user behaviour.
