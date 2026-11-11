"""Shared evaluation harness that every scorer passes through.

Metrics, stratified paired bootstrap with shared resample indices, the H1a to
H1c comparisons, H2 verdicts, the H3 horizon test, DeLong test and out of
fold cost thresholds. This is the only module that may read the test indices;
the test set is scored once, after the model freeze.

Owner: Abdalla (changes need Abdalla; Cameron as authorised deputy).
"""
