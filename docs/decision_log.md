# Decision log

Decisions are discussed in the group, decided by Abdalla and recorded here.
Hypothesis rules, margins, seed 42 and the Platt choice are committed here by
Gate 1 (Sun 15 Nov) and not altered after the test run.

| Date | Decision | Reason | Decided by | Commit |
| --- | --- | --- | --- | --- |
| 11 Nov 2026 | Python 3.11 (3.11.15 on the setup machine). Pinned runtime packages: torch 2.15.0 (CPU index), pandas 3.1.1, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, xgboost 3.2.0, shap 0.51.0, matplotlib 3.11.2, pyyaml 6.0.3, jupyter 1.1.1, ipykernel 7.4.0. Pinned dev tools: black 26.10.1, pytest 9.2.0, nbstripout 0.9.1. | Plan Section 4 fixes Python 3.11 and requires pinned requirements so the code runs from a clean environment on every member's machine. | Abdalla | TO FILL (project base commit) |
