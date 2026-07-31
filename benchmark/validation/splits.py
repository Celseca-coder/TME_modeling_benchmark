"""Train/test split strategies for patient-level spatial data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def patient_kfold(
    metadata: pd.DataFrame,
    n_folds: int = 5,
    patient_col: str = "patient_id",
    stratify_col: str | None = None,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Patient-level K-fold cross-validation splits.
    
        Assignments are made at the *patient* level to avoid data leakage when
        multiple regions share the same patient.
    
        Parameters
        ----------
        metadata : DataFrame with at least ``patient_col`` and ``region_id`` columns.
        n_folds : int
        patient_col : str
        stratify_col : str | None
            If given, stratify patient assignments by this column (e.g. the event
            indicator).  Value per patient is taken as the mode of that column.
        seed : int
    
        Returns
        -------
        list of (train_region_ids, val_region_ids) tuples, length = n_folds.
    
    Args:
        metadata (pd.DataFrame): Sample-level metadata used to construct data splits.
        n_folds (int): Number of folds.
        patient_col (str): Name of the column containing patient.
        stratify_col (str | None): Name of the column containing stratify.
        seed (int): Random seed used to make results reproducible."""
    # One row per patient
    if "region_id" not in metadata.columns:
        meta = metadata.reset_index(names="region_id")
    else:
        meta = metadata.copy()

    patient_df = meta.groupby(patient_col).agg(
        regions=("region_id", list)
    )

    if stratify_col and stratify_col in meta.columns:
        strat = meta.groupby(patient_col)[stratify_col].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        patient_df["_strat"] = strat
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_df, patient_df["_strat"])
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_df)

    splits = []
    patients = patient_df.index.tolist()
    all_regions = patient_df["regions"].tolist()

    for train_idx, val_idx in split_iter:
        train_regions = [r for i in train_idx for r in all_regions[i]]
        val_regions = [r for i in val_idx for r in all_regions[i]]
        splits.append((train_regions, val_regions))

    return splits


def stratify_column(task_cfg: dict) -> str | None:
    """The metadata column to stratify patient folds on, given a task config:
        the event indicator for survival, else the label column.
    
    Args:
        task_cfg (dict): Configuration mapping for the current prediction task."""
    if task_cfg["type"] == "survival":
        return task_cfg.get("event_col")
    return task_cfg.get("label_col")


def safe_patient_kfold(
    metadata: pd.DataFrame,
    n_folds: int,
    patient_col: str = "patient_id",
    stratify_col: str | None = None,
    seed: int = 0,
) -> list[tuple[list[str], list[str]]] | None:
    """:func:`patient_kfold` with graceful fallbacks for small cohorts.
    
        Caps ``n_folds`` at the number of patients, and drops stratification when a
        minority class is smaller than ``n_folds`` (which makes StratifiedKFold raise).
        Returns ``None`` when fewer than 2 folds are possible.
    
    Args:
        metadata (pd.DataFrame): Sample-level metadata used to construct data splits.
        n_folds (int): Number of folds.
        patient_col (str): Name of the column containing patient.
        stratify_col (str | None): Name of the column containing stratify.
        seed (int): Random seed used to make results reproducible."""
    n_pat = metadata[patient_col].nunique()
    folds = min(n_folds, n_pat)
    if folds < 2:
        return None
    try:
        return patient_kfold(metadata, n_folds=folds, patient_col=patient_col,
                             stratify_col=stratify_col, seed=seed)
    except ValueError:
        return patient_kfold(metadata, n_folds=folds, patient_col=patient_col,
                             stratify_col=None, seed=seed)


def cohort_split(
    metadata: pd.DataFrame,
    cohort_col: str = "cohort",
    train_values: list[str] | None = None,
    test_values: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split regions by cohort membership.
    
        Parameters
        ----------
        metadata     : DataFrame with ``region_id`` and ``cohort_col`` columns.
        cohort_col   : column identifying the cohort of each region.
        train_values : list of cohort values for training.
        test_values  : list of cohort values for testing.
    
        Returns
        -------
        (train_region_ids, test_region_ids)
    
    Args:
        metadata (pd.DataFrame): Sample-level metadata used to construct data splits.
        cohort_col (str): Name of the column containing cohort.
        train_values (list[str] | None): Values belonging to the candidate training samples.
        test_values (list[str] | None): Values belonging to the candidate test samples."""
    if "region_id" not in metadata.columns:
        meta = metadata.reset_index(names="region_id")
    else:
        meta = metadata.copy()

    if train_values is not None:
        train_mask = meta[cohort_col].isin(train_values)
    else:
        all_cohorts = meta[cohort_col].unique()
        test_set = set(test_values or [])
        train_mask = ~meta[cohort_col].isin(test_set)

    test_mask = ~train_mask if test_values is None else meta[cohort_col].isin(test_values)

    return (
        meta.loc[train_mask, "region_id"].tolist(),
        meta.loc[test_mask, "region_id"].tolist(),
    )
