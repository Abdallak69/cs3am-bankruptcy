"""Load the five UCI Polish bankruptcy ARFF files.

Reads each file from the raw data directory, decodes the class label
(1 = bankrupt), checks row counts, and removes exact duplicate rows from each
file before any split, logging the number removed. The five files are never
pooled.

Owner: Abdalla (project base).
"""
