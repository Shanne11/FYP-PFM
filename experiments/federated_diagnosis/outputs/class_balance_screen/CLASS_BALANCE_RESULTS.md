# Stage 3 Controlled Federated Class-Balanced Loss Experiment

## Configuration

Both variants used seed 42, the frozen BudgetWise training/validation split, 68 metadata features, the fixed 150-client map, all-client participation, architecture `68 → 128 → 64 → 13`, reset Adam at learning rate 0.001, three local epochs, batch size 32, 25 communication rounds, and sample-count FedAvg.

The initial global parameter SHA-256 was identical: `00a938248d2c12e2d14a7e1a58f3b04f4f6b78d7da061a3b25e7c455ebd8ed3d`.

Only the local training loss differed:

- **Standard:** unweighted cross-entropy.
- **Effective number:** class weight `w_c=(1-β)/(1-β^{n_c})`, with `β=0.9999`, using counts from the training partition only. The resulting 13 weights were divided by their arithmetic mean, so their mean equals 1.

Validation loss is unweighted standard cross-entropy for both configurations. The differently weighted local training-objective losses are recorded but are not interpreted as directly comparable quantities.

## Key-round results

| Variant | Round | Accuracy | Macro F1 | Weighted F1 | Coverage | Single-class collapse |
|---|---:|---:|---:|---:|---:|---|
| Standard | 4 | 0.2082 | 0.0401 | 0.0981 | 2 | No |
| Effective number | 4 | 0.1122 | 0.0540 | 0.0501 | 6 | No |
| Standard | 6 | 0.2169 | 0.0274 | 0.0773 | 1 | Yes |
| Effective number | 6 | 0.1255 | 0.0620 | 0.0591 | 6 | No |
| Standard | 10 | 0.2169 | 0.0274 | 0.0773 | 1 | Yes |
| Effective number | 10 | 0.1504 | 0.0785 | 0.0955 | 7 | No |
| Standard | 15 | 0.2169 | 0.0274 | 0.0773 | 1 | Yes |
| Effective number | 15 | 0.1781 | 0.0837 | 0.1189 | 7 | No |
| Standard | 20 | 0.2175 | 0.0284 | 0.0799 | 2 | No, but near-collapse |
| Effective number | 20 | 0.2186 | 0.0970 | 0.1402 | 6 | No |
| Standard | 25 | 0.2163 | 0.0309 | 0.0867 | 2 | No, but near-collapse |
| Effective number | 25 | 0.2296 | 0.0947 | 0.1457 | 6 | No |

## Peak and sustained performance

The predeclared model-selection rule is highest validation Macro F1, with round-25 results reported separately.

| Variant | Best round | Accuracy | Macro F1 | Weighted F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| Standard | 2 | 0.1463 | 0.0404 | 0.0811 | 5 |
| Effective number | 21 | 0.2273 | 0.0978 | 0.1450 | 6 |

At round 25, effective-number loss relative to standard loss:

- increases accuracy by 0.0133 absolute (1.33 percentage points);
- increases Macro F1 by 0.0638;
- increases Weighted F1 by 0.0590; and
- increases predicted-category coverage from 2 to 6.

The improvement is sustained rather than limited to an early checkpoint. Macro F1 remains between 0.0927 and 0.0978 across rounds 16–25, and coverage remains five to seven categories.

## Minority-category behaviour at round 25

Standard cross-entropy has non-zero recall only for Food (0.9920) and Rent (0.0067). Effective-number loss has non-zero recall for six categories:

| Category | Training count | Standard recall | Effective-number recall | Effective-number F1 |
|---|---:|---:|---:|---:|
| Bonus | 212 | 0.0000 | 0.4082 | 0.2469 |
| Education | 489 | 0.0000 | 0.0088 | 0.0175 |
| Food | 1,623 | 0.9920 | 0.7360 | 0.3812 |
| Investment | 227 | 0.0000 | 0.5000 | 0.2667 |
| Rent | 1,302 | 0.0067 | 0.0967 | 0.1352 |
| Travel | 956 | 0.0000 | 0.2036 | 0.1833 |

Minority recognition improves substantially for Bonus and Investment. However, seven categories—including Savings, the rarest training category—still have zero recall. Class balancing reduces collapse but does not solve the complete 13-category problem.

## Comparison with the centralised MLP

The previous three-seed centralised metadata-only MLP validation means were accuracy 0.2383 and Macro F1 0.1287. The seed-42 effective-number federated result at round 25 reaches accuracy 0.2296 and Macro F1 0.0947. It closes much of the accuracy gap, but remains approximately 0.0340 below the centralised Macro F1 mean.

This is not proof of parity because the central result is a three-seed mean while the federated result is one validation seed.

## Assessment

1. **Category collapse:** Prevented for this seed. Effective-number loss never predicts fewer than four categories and finishes with six.
2. **Round-25 Macro F1:** Improved from 0.0309 to 0.0947.
3. **Accuracy:** Improved by 1.33 percentage points rather than decreasing.
4. **Minority recall:** Improved for several categories, but remains zero for seven categories.
5. **Persistence:** Improvement is sustained through round 25.
6. **Replication:** Seeds 52 and 62 are justified, but require approval before execution.

The overall FYP categorisation problem is not solved by this single validation seed, and no held-out test evaluation was performed.
