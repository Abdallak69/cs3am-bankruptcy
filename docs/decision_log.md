# Decision log

Decisions are discussed in the group, decided by Abdalla and recorded here.
Hypothesis rules, margins, seed 42 and the Platt choice are committed here by
Gate 1 (Sun 15 Nov) and not altered after the test run.

| Date | Decision | Reason | Decided by | Commit |
| --- | --- | --- | --- | --- |
| 19 Nov 2026 | Exact duplicate rows (all 64 ratios and the label identical, with missing values in the same columns treated as equal) are removed from each file before the split, keeping the first occurrence. Removed: 1year 82, 2year 90, 3year 87, 4year 82, 5year 60. | Plan Sections 3 and 4. Duplicates on both sides of the split would leak test statements into training. | Abdalla | 5eb0ede |
| 19 Nov 2026 | Rows with identical ratios but different labels are kept and counted, not removed. None were found in any file. | No evidence favours one label over the other, so the rows are left as published. | Abdalla | 5eb0ede |
