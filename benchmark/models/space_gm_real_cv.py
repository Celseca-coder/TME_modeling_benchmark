"""Fast SPACE-GM cross-validation / cohort-generalization runner.

This mirrors the *logic* of :func:`benchmark.validation.cross_validate` and
:func:`benchmark.validation.cohort_split_test`, but builds each
:class:`spacegm.data.CellularGraphDataset` **once** and trains/evaluates every
fold by slicing ``train_inds`` / ``valid_inds`` — instead of re-featurizing and
re-processing the dataset on every fold (which dominates the run time).

Why this is equivalent (no leakage)
-----------------------------------
SPACE-GM's per-region featurization is deterministic and depends only on the
explicitly-passed vocabulary (``cell_type_mapping`` / ``cell_type_freq`` /
``biomarkers``) and fixed feature bounds — there are **no dataset-level
statistics** computed at construction (see the sanity check in the walkthrough
notebook). So a region's features are identical whether it sits in a per-fold
dataset or a shared one. Training restricts sampling to ``train_inds``
(``spacegm.data.SubgraphSampler(selected_inds=...)``) and inference restricts to
``valid_inds`` (``collect_predict_for_all_nodes(inds=...)``), both using
**absolute** region indices — so the validation regions never influence the
fitted model even though their graphs live in the same dataset.

* **cross-validation** — one dataset over all CV regions; a fresh model per fold.
* **cohort generalization** — cohorts use different cell-type columns, so *two*
  datasets are built (train cohort, test cohort) that **share** the vocabulary
  derived from the training cohort.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import spacegm as sg

from benchmark.features.space_gm_real import (
    SpaceGMGraphBuilder, NODE_FEATURES, EDGE_FEATURES, UNASSIGNED)
from benchmark.validation.metrics import score_predictions
from benchmark.validation.splits import safe_patient_kfold, stratify_column


@dataclass
class SpaceGMConfig:
    """Hyper-parameters for the SPACE-GM wrapper.

    Defaults are a deliberately light-weight version of the manuscript setup so
    the full benchmark is tractable; scale ``num_iterations`` / ``emb_dim`` up
    for a faithful reproduction (paper uses emb_dim=512, ~5e4 iterations).
    """
    # Graph / subgraph geometry (microns).
    subgraph_size: int = 3               # n-hop microenvironment
    subgraph_radius_limit_um: float = 75.0
    near_edge_um: float = 20.0
    subgraph_allow_distant_edge: bool = True

    # Model.
    emb_dim: int = 512
    gnn_type: str = "gin"
    graph_pooling: str = "max"
    drop_ratio: float = 0.25

    # Optimisation.
    num_iterations: int = 10000
    batch_size: int = 64
    lr: float = 1e-3
    graph_loss_weight: float = 1.0

    # Inference: fraction of a region's cells whose subgraphs are averaged.
    eval_subsample_ratio: float = 0.3

    # Feature processing (expression already z-scored by the featurizer).
    # By default only the cell-type identity is used (all other node features are
    # masked out via spacegm.transform.FeatureMask); add "biomarker_expression",
    # "neighborhood_composition", to use the full feature set.
    expression_clip: float = 3.0
    neighborhood_size: int = 10

    node_features: list = field(default_factory=lambda: list(NODE_FEATURES))
    edge_features: list = field(default_factory=lambda: list(EDGE_FEATURES))
    use_center_node_features: list[str] = field(default_factory=lambda: ["cell_type"])
    use_neighbor_node_features: list[str] = field(default_factory=lambda: ["cell_type"])

    device: str | None = None            # None -> cuda if available else cpu


# ======================================================================
# vocabulary + dataset construction
# ======================================================================
def derive_vocabulary(graphs: list):
    """Cell-type mapping / frequency and biomarker list from ``graphs``.

    ``Unassigned`` is index 0 so any remapped-unknown cell (at transform time on
    a held-out cohort) always resolves.
    """
    type_counts: dict[str, int] = {}
    biomarkers: set[str] = set()
    for G in graphs:
        for _, attrs in G.nodes(data=True):
            ct = attrs["cell_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1
            if not biomarkers and attrs.get("biomarker_expression"):
                biomarkers = set(attrs["biomarker_expression"].keys())
    ordered = [UNASSIGNED] + sorted(t for t in type_counts if t != UNASSIGNED)
    mapping = {ct: i for i, ct in enumerate(ordered)}
    total = max(sum(type_counts.values()), 1)
    freq = {ct: type_counts.get(ct, 0) / total for ct in ordered}
    freq[UNASSIGNED] = max(freq.get(UNASSIGNED, 0.0), 1e-6)
    return mapping, freq, sorted(biomarkers)


def build_dataset(regions, root, cfg, cell_type_col='cell_type', mapping=None, freq=None, biomarkers=None):
    """Write ``graphs`` and build a CellularGraphDataset (processed once).
    """
    print(f"Building dataset in {root} for {len(regions)} regions ...")
    raw_dir = os.path.join(root, "graph")
    os.makedirs(raw_dir, exist_ok=True)

    # ---- build nx graphs and cache ----
    builder = SpaceGMGraphBuilder(
        near_edge_um=cfg.near_edge_um, cell_type_col=cell_type_col).fit(regions)
    graphs = []
    for region in regions:
        g_path = os.path.join(raw_dir, f"{region.region_id}.gpkl")
        if os.path.exists(g_path):
            graph = pickle.load(open(g_path, "rb"))
        else:
            graph = builder.transform([region])['nx_graph'].iloc[0]
            with open(g_path, "wb") as f:
                pickle.dump(graph, f)
        graphs.append(graph)

    # ---- calculate dataset-wide vocabulary ----
    if mapping is None or freq is None or biomarkers is None:
        mapping, freq, biomarkers = derive_vocabulary(graphs)

    # ---- build dataset ----
    kwargs = dict(
        pre_transform=None,
        raw_folder_name="graph",
        processed_folder_name="tg_graph",
        node_features=list(cfg.node_features),
        edge_features=list(cfg.edge_features),
        cell_type_mapping=mapping,
        cell_type_freq=freq,
        biomarkers=list(biomarkers),
        subgraph_size=cfg.subgraph_size,
        subgraph_source="on-the-fly",
        subgraph_allow_distant_edge=cfg.subgraph_allow_distant_edge,
        subgraph_radius_limit=cfg.subgraph_radius_limit_um,
        biomarker_expression_process_method="linear",
        biomarker_expression_lower_bound=-cfg.expression_clip,
        biomarker_expression_upper_bound=cfg.expression_clip,
        neighborhood_size=cfg.neighborhood_size,
    )
    dataset = sg.CellularGraphDataset(root, **kwargs)
    return dataset


# ======================================================================
# model train / predict on explicit indices
# ======================================================================
def _new_gnn(dataset, cfg, num_graph_tasks):
    return sg.models.GNN_pred(
        num_layer=cfg.subgraph_size,
        num_node_type=len(dataset.cell_type_mapping) + 1,
        num_feat=dataset[0].x.shape[1] - 1,
        emb_dim=cfg.emb_dim,
        num_node_tasks=0,
        num_graph_tasks=num_graph_tasks,
        node_embedding_output="last",
        drop_ratio=cfg.drop_ratio,
        graph_pooling=cfg.graph_pooling,
        gnn_type=cfg.gnn_type,
    )


def train_on_inds(dataset, cfg, seed, num_graph_tasks, loss_fn, train_inds, model_dir=None):
    """Fresh model trained on ``train_inds`` (absolute region indices)."""
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = _new_gnn(dataset, cfg, num_graph_tasks)

    model = sg.train.train_subgraph(
        model, dataset, device,
        train_inds=np.asarray(train_inds),
        valid_inds=None,
        num_iterations=cfg.num_iterations,
        num_regions_per_segment=0,
        num_iterations_per_segment=max(int(cfg.num_iterations) * 10, 1),
        evaluate_freq=max(int(cfg.num_iterations // 10), 500),
        evaluate_fn=[sg.train.save_model_weight] if model_dir else [],
        evaluate_on_train=False,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        graph_loss_weight=cfg.graph_loss_weight,
        graph_task_loss_fn=loss_fn,
        model_folder=model_dir,
    )
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model_final.pt"))
        with open(os.path.join(model_dir, "vocab.json"), "w") as f:
            json.dump({"cell_type_mapping": dataset.cell_type_mapping,
                       "num_graph_tasks": int(num_graph_tasks),
                       "emb_dim": int(cfg.emb_dim),
                       "subgraph_size": int(cfg.subgraph_size)}, f, indent=2)
    return model


def predict_on_inds(model, dataset, cfg, inds) -> dict:
    """``region_id -> mean graph-head output`` over sampled subgraphs of ``inds``."""
    import torch
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, graph_results = sg.inference.collect_predict_for_all_nodes(
        model, dataset, device,
        inds=np.asarray(inds),
        batch_size=cfg.batch_size,
        shuffle=True,
        subsample_ratio=cfg.eval_subsample_ratio,
    )
    out = {}
    for i, preds in graph_results.items():
        valid = [p for p in preds if p is not None and np.all(p == p)]
        out[i] = valid
    return out


# ======================================================================
# task specs (label transform + loss + prediction post-processing)
# ======================================================================
def _classification_spec(target: pd.Series, root: str, label_file_name: str = None):
    if label_file_name is None:
        label_file_name = "graph_labels.csv"
    classes = np.sort(pd.unique(target))
    label_path = os.path.join(root, label_file_name)

    if len(classes) > 2:
        raise
    else:
        label_cols = ["label"]
        label_df = pd.DataFrame(
            {"label": (target.to_numpy() == classes[-1]).astype(int)},
            index=[str(i) for i in target.index])
        label_df.index.name = "REGION_ID"
        num_tasks = 1
    label_df.to_csv(label_path)

    transform = sg.transform.AddGraphLabel(label_path, tasks=list(label_cols))
    loss_fn = sg.models.BinaryCrossEntropy()

    def postprocess(pred_map, index):
        agg_pred_vals = [np.median(np.array(pred_map[i]).flatten()) if i in pred_map else np.nan for i in index]
        p1 = np.array(agg_pred_vals)
        p1 = 1.0 / (1.0 + np.exp(-p1))
        return np.column_stack([1.0 - p1, p1])

    return transform, loss_fn, num_tasks, classes, postprocess


class _AddSurvivalLabel:
    """SPACE-GM transform attaching survival labels (time -> graph_y, event -> graph_w).

    This mirrors :class:`spacegm.transform.AddGraphLabel` but stores the two
    survival channels in the ``graph_y`` / ``graph_w`` slots consumed by
    ``spacegm.models.CoxSGDLossFn(y_pred, length, event)`` inside
    ``spacegm.train.train_subgraph``.
    """

    def __init__(self, label_csv: str):
        import torch  # noqa: F401 (kept local to avoid importing torch at module load)
        df = pd.read_csv(label_csv, index_col=0)
        df.index = df.index.map(str)
        self._time = df["time"].astype(float).to_dict()
        self._event = df["event"].astype(float).to_dict()

    def __call__(self, data):
        import torch
        rid = str(data.region_id)
        data.graph_y = torch.tensor([[self._time[rid]]], dtype=torch.float32)
        data.graph_w = torch.tensor([[self._event[rid]]], dtype=torch.float32)
        return data


def _survival_spec(target: pd.DataFrame, root: str, label_file_name: str = None):
    if label_file_name is None:
        label_file_name = "survival_labels.csv"
    label_path = os.path.join(root, label_file_name)
    label_df = pd.DataFrame(
        {"time": target["time"].astype(float).to_numpy(),
         "event": target["event"].astype(float).to_numpy()},
        index=[str(i) for i in target.index])
    label_df.index.name = "REGION_ID"
    label_df.to_csv(label_path)

    transform = _AddSurvivalLabel(label_path)
    loss_fn = sg.models.CoxSGDLossFn(top_n=0, regularizer_weight=1e-3)

    def postprocess(pred_map, index):
        agg_pred_vals = [np.median(np.array(pred_map[i]).flatten()) if i in pred_map else np.nan for i in index]
        return np.array(agg_pred_vals)

    return transform, loss_fn, 1, None, postprocess


def _make_spec(task_cfg: dict, target, root: str, label_file_name: str = None):
    if task_cfg["type"] == "survival":
        return _survival_spec(target, root, label_file_name=label_file_name)
    return _classification_spec(target, root, label_file_name=label_file_name)


def _nan_metrics(task_type: str) -> dict:
    if task_type == "survival":
        return {"c_index": float("nan")}
    if task_type in ("binary", "binary_classification"):
        return {"auc_roc": float("nan"), "avg_precision": float("nan"),
                "balanced_acc": float("nan")}
    return {"balanced_acc": float("nan"), "macro_auc": float("nan")}


# ======================================================================
# orchestration
# ======================================================================
def cross_validate_fast(ds, task_id, cfg: SpaceGMConfig, *, seeds=(0, 1, 2),
                        model_dir=None, work_root=None, normalize=True) -> list[dict]:
    """Patient-level K-fold CV with a single, once-processed dataset.

    Returns one metric dict per (seed, fold), matching
    :func:`benchmark.validation.cross_validate`'s output shape.
    """
    vcfg = ds.validation_config
    n_folds = vcfg.get("n_folds", 5)
    patient_col = vcfg.get("patient_col", "patient_id")
    cv_filter = vcfg.get("cv_filter")
    task_cfg = ds.get_task_config(task_id)

    meta = ds.get_task_metadata(task_id)
    if len(meta) == 0:
        return []
    if cv_filter:
        sub = meta.query(cv_filter)
        meta = sub if len(sub) else meta

    region_ids = meta["region_id"].tolist()
    regions = ds.load_regions(region_ids, normalize=normalize)

    # ---- Define and build dataset ----
    if work_root is None:
        work_root = tempfile.TemporaryDirectory(prefix="sgm_cv_", dir=None).name
    os.makedirs(work_root, exist_ok=True)
    dataset_root = os.path.join(work_root, "ds")
    dataset = build_dataset(regions, dataset_root, cfg)

    # ---- Build label file and label transform ----
    target_all = ds.build_target(region_ids, task_id)
    label_transform, loss_fn, num_tasks, classes, postprocess = \
        _make_spec(task_cfg, target_all, work_root)

    # ---- Attach transform ----
    transforms = [sg.transform.FeatureMask(
        dataset,
        use_center_node_features=cfg.use_center_node_features,
        use_neighbor_node_features=cfg.use_neighbor_node_features),
        label_transform]
    dataset.transform = transforms
    ridx = {str(r): i for i, r in enumerate(dataset.region_ids)}

    # ---- Run CV ----
    fold_metrics: list[dict] = []
    for seed in seeds:
        folds = safe_patient_kfold(meta, n_folds, patient_col, stratify_column(task_cfg), seed)
        if folds is None:
            continue
        for fold_i, (train_ids, val_ids) in enumerate(folds):
            tr_inds = [ridx[str(r)] for r in train_ids if str(r) in ridx]
            va_inds = [ridx[str(r)] for r in val_ids if str(r) in ridx]
            try:
                model_folder = os.path.join(
                    model_dir,
                    f"seed{seed}_fold{fold_i}_{uuid.uuid4().hex[:8]}")

                model = train_on_inds(
                    dataset, cfg, seed, num_tasks, loss_fn, tr_inds,
                    model_dir=model_folder)
                pred_map = predict_on_inds(model, dataset, cfg, va_inds)

                y_va = target_all.loc[[dataset.region_ids[i] for i in va_inds]]
                y_pred = postprocess(pred_map, va_inds)
                metrics = score_predictions(task_cfg, y_va, y_pred, classes)
            except Exception:
                if os.environ.get("BENCHMARK_RAISE_ERRORS"):
                    raise
                metrics = _nan_metrics(task_cfg["type"])
            metrics.update({"seed": seed, "fold": fold_i,
                            "n_train": len(tr_inds), "n_val": len(va_inds)})
            fold_metrics.append(metrics)
            pd.DataFrame(fold_metrics).to_csv(os.path.join(work_root, "cv_metrics.csv"), index=False)

    return fold_metrics


def cohort_generalization_fast(ds, task_id, gentest, cfg: SpaceGMConfig, *,
                               seeds=(0,), model_dir=None, work_root=None,
                               normalize=True) -> list[dict]:
    """Train on the train cohort, evaluate on the test cohort (two datasets).

    Returns one metric dict per seed, matching
    :func:`benchmark.validation.cohort_split_test`'s output shape.
    """
    cohort_col = ds.validation_config["cohort_col"]
    task_cfg = ds.get_task_config(task_id)
    meta = ds.get_task_metadata(task_id)

    train_ids = meta[meta[cohort_col].isin(gentest["train"])]["region_id"].tolist()
    test_ids = meta[meta[cohort_col].isin(gentest["test"])]["region_id"].tolist()
    if not train_ids or not test_ids:
        return []
    cell_type_col = gentest.get("cell_type_col", "cell_type")

    tr_regions = ds.load_regions(train_ids, normalize=normalize)
    te_regions = ds.load_regions(test_ids, normalize=normalize)

    # ---- Define and build dataset ----
    if work_root is None:
        work_root = tempfile.TemporaryDirectory(prefix="sgm_cv_", dir=None).name
    os.makedirs(work_root, exist_ok=True)
    ds_tr_root = os.path.join(work_root, "ds_tr")
    dataset_tr = build_dataset(tr_regions, ds_tr_root, cfg, cell_type_col=cell_type_col)

    ds_te_root = os.path.join(work_root, "ds_te")
    dataset_te = build_dataset(
        te_regions, ds_te_root, cfg, cell_type_col=cell_type_col,
        mapping=dataset_tr.cell_type_mapping, freq=dataset_tr.cell_type_freq,
        biomarkers=dataset_tr.biomarkers)

    assert dataset_tr.cell_type_mapping == dataset_te.cell_type_mapping
    assert dataset_tr.node_feature_names == dataset_te.node_feature_names

    # ---- Build label file and label transform ----
    y_tr = ds.build_target([r.region_id for r in tr_regions], task_id)
    y_te = ds.build_target([r.region_id for r in te_regions], task_id)

    label_transform_tr, loss_fn, num_tasks, classes, postprocess = \
        _make_spec(task_cfg, y_tr, work_root, label_file_name="graph_labels_tr.csv")
    label_transform_te, _, _, _, _ = \
        _make_spec(task_cfg, y_te, work_root, label_file_name="graph_labels_te.csv")

    # ---- Attach transform ----
    input_transforms = [sg.transform.FeatureMask(
        dataset_tr,
        use_center_node_features=cfg.use_center_node_features,
        use_neighbor_node_features=cfg.use_neighbor_node_features)]
    dataset_tr.transform = input_transforms + [label_transform_tr]
    dataset_te.transform = input_transforms + [label_transform_te]


    run_metrics: list[dict] = []
    for seed in seeds:
        for run_i in range(3):
            try:
                model_folder = os.path.join(
                    model_dir,
                    f"seed{seed}_run{run_i}_{uuid.uuid4().hex[:8]}")

                model = train_on_inds(
                    dataset_tr, cfg, seed, num_tasks, loss_fn, np.arange(dataset_tr.N),
                    model_dir=model_folder)

                pred_map = predict_on_inds(model, dataset_te, cfg, np.arange(dataset_te.N))

                _y_te = y_te.loc[[dataset_te.region_ids[i] for i in np.arange(dataset_te.N)]]
                y_pred = postprocess(pred_map, np.arange(dataset_te.N))
                metrics = score_predictions(task_cfg, _y_te, y_pred, classes)
            except Exception:
                if os.environ.get("BENCHMARK_RAISE_ERRORS"):
                    raise
                metrics = _nan_metrics(task_cfg["type"])
            metrics.update({"seed": seed, "run": run_i,
                            "n_train": dataset_tr.N, "n_val": dataset_te.N})
            run_metrics.append(metrics)
            pd.DataFrame(run_metrics).to_csv(os.path.join(work_root, "cv_metrics.csv"), index=False)

    return run_metrics
