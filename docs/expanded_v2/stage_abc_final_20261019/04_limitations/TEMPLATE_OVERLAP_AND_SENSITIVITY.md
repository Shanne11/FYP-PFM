# Description-template overlap and sensitivity

Participant separation is preserved, but description-template separation is not. A test record is labelled **overlapping** when its frozen `description_group_id` appears in that seed's training or validation partition. This is a sensitivity analysis of frozen predictions; models were not refit.

Status: **Completed as a descriptive sensitivity**.

## Overlap counts

| seed | train_records | validation_records | test_records | train_groups | validation_groups | test_groups | train_validation_shared_groups | train_test_shared_groups | validation_test_shared_groups | test_groups_seen_in_train_or_validation | test_records_with_prior_template | test_records_without_prior_template | description_template_disjoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 176 | 35 | 43 | 108 | 27 | 35 | 14 | 14 | 10 | 16 | 24 | 19 | False |
| 52 | 178 | 35 | 41 | 108 | 29 | 28 | 11 | 14 | 4 | 14 | 27 | 14 | False |
| 62 | 178 | 32 | 44 | 103 | 27 | 32 | 9 | 12 | 3 | 13 | 24 | 20 | False |
| 72 | 178 | 32 | 44 | 113 | 25 | 29 | 11 | 15 | 8 | 16 | 31 | 13 | False |
| 82 | 177 | 35 | 42 | 100 | 26 | 35 | 11 | 10 | 4 | 10 | 16 | 26 | False |

## Frozen-prediction sensitivity

| seed | template_subset | model | records | participants | accuracy | macro_f1_fixed_14 | Wrong->Correct | Correct->Wrong | Correct->Correct | Wrong->Wrong | net_corrections |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | non_overlapping | B1 | 19 | 5 | 0.6842 | 0.1581 | 3 | 0 | 13 | 3 | 3 |
| 42 | non_overlapping | B2 | 19 | 5 | 0.8421 | 0.2012 | 3 | 0 | 13 | 3 | 3 |
| 42 | overlapping | B1 | 24 | 5 | 0.7500 | 0.3670 | 3 | 1 | 17 | 3 | 2 |
| 42 | overlapping | B2 | 24 | 5 | 0.8333 | 0.3755 | 3 | 1 | 17 | 3 | 2 |
| 52 | non_overlapping | B1 | 14 | 5 | 0.6429 | 0.1259 | 3 | 0 | 9 | 2 | 3 |
| 52 | non_overlapping | B2 | 14 | 5 | 0.8571 | 0.2786 | 3 | 0 | 9 | 2 | 3 |
| 52 | overlapping | B1 | 27 | 5 | 0.7407 | 0.2267 | 3 | 0 | 20 | 4 | 3 |
| 52 | overlapping | B2 | 27 | 5 | 0.8519 | 0.2843 | 3 | 0 | 20 | 4 | 3 |
| 62 | non_overlapping | B1 | 20 | 5 | 0.6500 | 0.1542 | 2 | 0 | 13 | 5 | 2 |
| 62 | non_overlapping | B2 | 20 | 5 | 0.7500 | 0.2487 | 2 | 0 | 13 | 5 | 2 |
| 62 | overlapping | B1 | 24 | 5 | 0.5417 | 0.1199 | 5 | 0 | 13 | 6 | 5 |
| 62 | overlapping | B2 | 24 | 5 | 0.7500 | 0.1618 | 5 | 0 | 13 | 6 | 5 |
| 72 | non_overlapping | B1 | 13 | 5 | 0.3846 | 0.1190 | 2 | 0 | 5 | 6 | 2 |
| 72 | non_overlapping | B2 | 13 | 5 | 0.5385 | 0.1718 | 2 | 0 | 5 | 6 | 2 |
| 72 | overlapping | B1 | 31 | 5 | 0.6129 | 0.1859 | 5 | 0 | 19 | 7 | 5 |
| 72 | overlapping | B2 | 31 | 5 | 0.7742 | 0.2399 | 5 | 0 | 19 | 7 | 5 |
| 82 | non_overlapping | B1 | 26 | 5 | 0.7308 | 0.2861 | 2 | 0 | 19 | 5 | 2 |
| 82 | non_overlapping | B2 | 26 | 5 | 0.8077 | 0.3621 | 2 | 0 | 19 | 5 | 2 |
| 82 | overlapping | B1 | 16 | 5 | 0.5625 | 0.2133 | 4 | 0 | 9 | 3 | 4 |
| 82 | overlapping | B2 | 16 | 5 | 0.8125 | 0.3131 | 4 | 0 | 9 | 3 | 4 |

Component-disjoint sensitivity is infeasible because the participant/description graph contains one connected component. No substitute split was generated.
