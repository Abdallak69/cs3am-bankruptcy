"""Baseline scorers: original Altman Z', refitted Z' and logistic regression.

Original Z' uses the published coefficients on pipeline output, with -Z' as
the risk score. Refitted Z' is a logistic regression on the five Altman
inputs only. The statistical baseline is a logistic regression on all 64
ratios. Both logistic regressions use standardised inputs, L2 penalty,
balanced class weights and C tuned with the shared cross validation.

Owner: Abdalla.
"""
