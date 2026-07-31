"""Expose validation splits, scoring helpers, and benchmark runners."""

from .splits import patient_kfold, cohort_split
from .metrics import score_predictions, PRIMARY_METRIC, summarize_folds
from .crossval import cross_validate, cohort_split_test
