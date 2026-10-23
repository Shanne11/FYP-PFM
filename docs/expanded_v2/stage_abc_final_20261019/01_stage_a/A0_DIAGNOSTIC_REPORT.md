# A0 Diagnostic Report

Status: independently recomputed from the preserved split manifests and A0 test
predictions. No model was retrained.

Across all five seeds A0 predicted the training-majority label for every test
record and the saved prediction labels aligned with the split manifests. The
direct arithmetic matched every saved accuracy value. The mean training share of
the majority label was 27.26%, whereas its mean
test share was only 0.04%. The near-zero result is
therefore an expected consequence of template-group-disjoint distribution shift:
the description groups carrying the training-majority class are concentrated in
training and are almost absent from some test partitions. No label-index or A0
implementation defect was detected.

Evidence: `A0_DIAGNOSTIC.csv` and `A0_GROUP_COUNTS.csv`.
