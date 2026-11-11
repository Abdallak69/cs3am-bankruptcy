"""Frozen stratified 80/20 train/test split.

Creates one stratified split per file with the nominated seed (42), after
duplicate removal, and writes the test indices once to data/splits/. The
5year.arff split is reused by every experiment. Apart from evaluate.py, this
is the only module that touches the test indices, and it only writes them.

Owner: Abdalla (changes need Abdalla; Cameron as authorised deputy).
"""
