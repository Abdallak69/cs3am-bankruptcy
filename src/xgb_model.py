"""Model 1: XGBoost.

Random search with tuned class weighting, early stopping on the inner
validation split of each training fold, final refit at the median best
iteration from cross validation, and final models trained on the seeds in
config.yaml.

Owner: Cameron.
"""
