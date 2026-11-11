"""Preprocessing pipeline (sklearn Pipeline), fitted on training data only.

Steps: infinite values to missing; missingness indicator for each ratio with
more than 5% missing; median imputation; winsorising at the 1st and 99th
percentiles of the training fold; dropping one of each pair with absolute
Spearman correlation above 0.95 while never dropping the five Altman inputs
(Attr3, Attr6, Attr7, Attr8, Attr9); standardising for both logistic
regressions and the MLP (or a quantile transform for the MLP, decided on CV
evidence).

Owner: Cameron.
"""
