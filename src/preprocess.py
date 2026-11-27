"""Preprocessing of the 64 ratios, fitted on training data only.

RatioPreprocessor works through the steps of the project plan in order:
infinite values to missing; a missingness indicator for each ratio with more
than 5% missing; median imputation; winsorising at the 1st and 99th
percentiles; and dropping one of each pair of ratios with absolute Spearman
correlation above 0.95, keeping the less missing ratio and never dropping one
of the five Altman inputs (Attr3, Attr6, Attr7, Attr8, Attr9).

Every number it uses (missing shares, medians, percentiles, correlations) is
learned in fit, so the baselines and models fit it on the fitting rows of each
fold only. It does not standardise: the original Z' needs the ratios on their
original scale, so each scorer adds its own scaler (a standard scaler for the
logistic regressions; the MLP's own pipeline for the MLP).

Owner: Cameron.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Altman, E. I. (1983), Corporate Financial Distress, Wiley: Z' inputs X1 to X5
# as the UCI ratios Attr3, Attr6, Attr7, Attr8 and Attr9. The same list lives
# in baselines.py; tests/test_preprocess.py checks that the two agree.
ALTMAN_INPUTS: tuple[str, ...] = ("Attr3", "Attr6", "Attr7", "Attr8", "Attr9")

INDICATOR_SUFFIX: str = "_missing"
MISSING_SHARE: float = 0.05
LOWER_QUANTILE: float = 0.01
UPPER_QUANTILE: float = 0.99
MAX_ABS_SPEARMAN: float = 0.95


def infinite_to_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with NaN.

    Args:
        frame: Rows of ratios.

    Returns:
        A copy of frame in which every infinite value is NaN.
    """
    return frame.replace([np.inf, -np.inf], np.nan)


def ratios_to_keep(
    winsorised: pd.DataFrame,
    missing: pd.Series,
    protected: tuple[str, ...],
    max_correlation: float,
) -> list[str]:
    """Choose the ratios left after dropping one of each highly correlated pair.

    Pairs are checked in column order. Of a pair with absolute Spearman
    correlation above max_correlation, the ratio with the larger missing
    share is dropped, and on a tie the later column. A protected ratio is
    never dropped, so its partner goes instead; two protected ratios both
    stay.

    Args:
        winsorised: Imputed and winsorised training ratios.
        missing: Share of each ratio that was missing in the training rows
            before imputation, indexed by ratio name.
        protected: Ratios that are never dropped.
        max_correlation: Largest absolute Spearman correlation allowed.

    Returns:
        The names of the ratios to keep, in their original order.
    """
    names = list(winsorised.columns)
    # A constant ratio has an undefined correlation, so count it as zero.
    correlation = winsorised.corr(method="spearman").abs().fillna(0.0).to_numpy()

    dropped: set[str] = set()
    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            second = names[j]
            if first in dropped or second in dropped:
                continue
            if correlation[i, j] <= max_correlation:
                continue
            if first in protected and second in protected:
                continue
            if first in protected:
                dropped.add(second)
            elif second in protected:
                dropped.add(first)
            elif missing[first] > missing[second]:
                dropped.add(first)
            else:
                dropped.add(second)
    return [name for name in names if name not in dropped]


class RatioPreprocessor(BaseEstimator, TransformerMixin):
    """Clean the ratios; takes and returns a DataFrame with column names kept.

    Attributes learned in fit: ratio_names_, flagged_, medians_, lower_,
    upper_ and kept_.
    """

    def __init__(
        self,
        missing_share: float = MISSING_SHARE,
        lower_quantile: float = LOWER_QUANTILE,
        upper_quantile: float = UPPER_QUANTILE,
        max_correlation: float = MAX_ABS_SPEARMAN,
        protected: tuple[str, ...] = ALTMAN_INPUTS,
    ):
        """Store the settings; nothing is learned until fit.

        Args:
            missing_share: A ratio gets a missingness indicator if more than
                this share of the fitting rows lack it.
            lower_quantile: Lower winsorising percentile, as a fraction.
            upper_quantile: Upper winsorising percentile, as a fraction.
            max_correlation: Largest absolute Spearman correlation allowed
                between two kept ratios.
            protected: Ratios that are never dropped.
        """
        self.missing_share = missing_share
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.max_correlation = max_correlation
        self.protected = protected

    def fit(self, X: pd.DataFrame, y: object = None) -> "RatioPreprocessor":
        """Learn the missing shares, medians, percentiles and ratios to keep.

        Args:
            X: Fitting rows, one column per ratio. Must not include the
                rows a model is later scored on.
            y: Ignored; the preprocessing does not use the labels.

        Returns:
            This preprocessor.
        """
        ratios = infinite_to_missing(X)
        self.ratio_names_ = list(ratios.columns)

        missing = ratios.isna().mean()
        self.flagged_ = [
            n for n in self.ratio_names_ if missing[n] > self.missing_share
        ]

        # A ratio that is missing everywhere has no median; 0.0 stands in.
        self.medians_ = ratios.median().fillna(0.0)
        imputed = ratios.fillna(self.medians_)
        self.lower_ = imputed.quantile(self.lower_quantile)
        self.upper_ = imputed.quantile(self.upper_quantile)
        winsorised = imputed.clip(self.lower_, self.upper_, axis=1)

        self.kept_ = ratios_to_keep(
            winsorised, missing, tuple(self.protected), self.max_correlation
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean rows with the numbers learned in fit.

        Args:
            X: Rows with the same ratio columns as in fit.

        Returns:
            The kept ratios, imputed and winsorised, on their original scale
            and keeping their names and index, followed by one 0/1 column
            named <ratio>_missing for each flagged ratio.
        """
        ratios = infinite_to_missing(X)[self.ratio_names_]
        indicators = ratios[self.flagged_].isna().astype(float)
        indicators = indicators.add_suffix(INDICATOR_SUFFIX)
        winsorised = ratios.fillna(self.medians_).clip(self.lower_, self.upper_, axis=1)
        return pd.concat([winsorised[self.kept_], indicators], axis=1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Name the columns transform returns.

        Args:
            input_features: Ignored; the names are learned in fit.

        Returns:
            The kept ratios followed by the missingness indicators.
        """
        names = self.kept_ + [name + INDICATOR_SUFFIX for name in self.flagged_]
        return np.array(names, dtype=object)


def make_preprocessor() -> RatioPreprocessor:
    """Return a new unfitted preprocessor, for baselines.py and the models.

    The preprocessor takes and returns a DataFrame with column names kept,
    never drops or renames Attr3, Attr6, Attr7, Attr8 or Attr9, and does not
    standardise.

    Returns:
        A RatioPreprocessor set to output pandas DataFrames.
    """
    return RatioPreprocessor().set_output(transform="pandas")
