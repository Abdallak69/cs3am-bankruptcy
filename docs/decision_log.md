# Decision log

Decisions are discussed in the group, decided by Abdalla and recorded here.
Hypothesis rules, margins, seed 42 and the Platt choice are committed here by
Gate 1 (Sun 15 Nov) and not altered after the test run.

| Date | Decision | Reason | Decided by | Commit |
| --- | --- | --- | --- | --- |
| 19 Nov 2026 | Exact duplicate rows (all 64 ratios and the label identical, with missing values in the same columns treated as equal) are removed from each file before the split, keeping the first occurrence. Removed: 1year 82, 2year 90, 3year 87, 4year 82, 5year 60. | Plan Sections 3 and 4. Duplicates on both sides of the split would leak test statements into training. | Abdalla | 5eb0ede |
| 19 Nov 2026 | Rows with identical ratios but different labels are kept and counted, not removed. None were found in any file. | No evidence favours one label over the other, so the rows are left as published. | Abdalla | 5eb0ede |
| 19 Nov 2026 | Each deduplicated file gets one stratified 80/20 split on the class label with seed 42, written once to data/splits/ as sorted training and test row_ids. Training rows (bankrupt) / test rows (bankrupt): 1year 5,556 (217) / 1,389 (54); 2year 8,066 (318) / 2,017 (80); 3year 8,332 (394) / 2,084 (99); 4year 7,768 (410) / 1,942 (103); 5year 4,680 (326) / 1,170 (82). | Plan Section 4. The split is frozen before any modelling, and write_splits refuses to overwrite it. | Abdalla | 09e83f1 |
| 19 Nov 2026 | Training row_ids are saved alongside the test row_ids, and every module other than evaluate.py loads data through load_training_set. No other module or notebook may refer to the test indices or to split_file, which can rebuild them; the test indices guard checks src/ and notebooks/. | Code outside evaluate.py never needs to know which rows are in the test set. | Abdalla | 09e83f1 |
