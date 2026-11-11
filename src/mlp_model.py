"""Model 2: PyTorch MLP (deep learning).

Dataset class, training loop with BCEWithLogitsLoss and AdamW, early stopping
on inner validation split PR AUC, seed control, random search with tuned
class weighting, final refit at the median best epoch, and final models
trained on the seeds in config.yaml.

Owner: Zack.
"""
