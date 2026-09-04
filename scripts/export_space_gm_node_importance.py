#!/usr/bin/env python
"""Export per-cell importances from trained SPACE-GM checkpoints.

Reads ``model_results/SPACE_GM_pseudo/{dataset}_{task}_cv/`` (noisy v2 labels)
and writes ``region_id,cell_type,importance,task`` for
``verify_pseudo_label_explanations.py --mode gnn-explainer``.

Methods (all applied after the training FeatureMask that keeps only cell type):

* ``gnn-explainer``: post-embedding node mask (SPACE-GM's native hook; does not
  multiply the integer cell-type id).
* ``ig``: integrated gradients of the positive logit w.r.t. that node mask.
* ``occlusion``: drop in logit after zeroing one node's embedding.

The trained model is official SPACE-GM (GIN, max pool, 3-hop, emb_dim=512),
not ``benchmark.models.space_gm.SpaceGMClassifier``. Run this in the
``space_gm`` env.

    python scripts/export_space_gm_node_importance.py \\
        --dataset hnc_wu2022 \\
        --tasks motif_cd8_clustering motif_immune_exclusion \\
        --method gnn-explainer --device cuda
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

import spacegm as sg  # noqa: E402
from spacegm.features import get_feature_names  # noqa: E402

NODE_FEATURES = [
    "cell_type",
    "biomarker_expression",
    "neighborhood_composition",
    "center_coord",
]
EDGE_FEATURES = ["edge_type", "distance"]
SELECTED = {
    "bc_jackson2020": [
        "motif_tumor_high",
        "motif_t_tumor_mixing",
        "motif_cd8_tumor_contact",
        "motif_macrophage_tumor_niche",
        "motif_apc_t_contact",
    ],
    "hnc_wu2022": ["motif_cd8_clustering", "motif_immune_exclusion"],
    "bc_metabric_ali2020": ["motif_tumor_stroma_mixing", "motif_interface_immune"],
    "tnbc_wang2023": ["motif_cd8_high"],
}


class _FeatHolder:
    def __init__(self, node_feature_names, edge_feature_names):
        self.node_feature_names = list(node_feature_names)
        self.edge_feature_names = list(edge_feature_names)
        self.node_features = list(NODE_FEATURES)
        self.edge_features = list(EDGE_FEATURES)


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _dataset_cfg(dataset: str) -> dict:
    return _load_yaml(_CODE / "configs" / "datasets" / f"{dataset}.yaml")


def _catalog_cfg(dataset: str) -> dict:
    return _load_yaml(_CODE / "configs" / "motifs" / f"{dataset}.yaml")


def _reverse_mapping(mapping: dict) -> dict[int, str]:
    return {int(i): str(name) for name, i in mapping.items()}


def _biomarkers_from_graph(graph_dir: Path) -> list[str]:
    gpkl = next(graph_dir.glob("*.gpkl"))
    graph = pickle.load(gpkl.open("rb"))
    node = next(iter(graph.nodes))
    expr = graph.nodes[node].get("biomarker_expression") or {}
    return sorted(expr)


def _feature_mask(node_feature_names, edge_feature_names):
    holder = _FeatHolder(node_feature_names, edge_feature_names)
    return sg.transform.FeatureMask(
        holder,
        use_center_node_features=["cell_type"],
        use_neighbor_node_features=["cell_type"],
    )


def _apply_mask(data, mask_fn):
    from copy import deepcopy
    return mask_fn(deepcopy(data))


def _load_model(ckpt: Path, vocab: dict, num_feat: int, device: str):
    mapping = vocab["cell_type_mapping"]
    model = sg.models.GNN_pred(
        num_layer=int(vocab.get("subgraph_size", 3)),
        num_node_type=len(mapping) + 1,
        num_feat=int(num_feat),
        emb_dim=int(vocab.get("emb_dim", 512)),
        num_node_tasks=0,
        num_graph_tasks=int(vocab.get("num_graph_tasks", 1)),
        node_embedding_output="last",
        drop_ratio=0.25,
        graph_pooling="max",
        gnn_type="gin",
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _logit(model, data, node_feat_mask=None):
    out = model(data, node_feat_mask=node_feat_mask)
    pred = out[-1] if isinstance(out, (list, tuple)) else out
    return pred.view(-1)[0]


def _gnn_explainer(model, data, device, epochs: int, lr: float) -> torch.Tensor:
    n = int(data.num_nodes)
    data = data.to(device)
    mask = torch.nn.Parameter(0.01 * torch.randn(n, 1, device=device))
    opt = torch.optim.Adam([mask], lr=lr)
    with torch.no_grad():
        target = (_logit(model, data) > 0).float()
    for _ in range(epochs):
        opt.zero_grad()
        m = torch.sigmoid(mask)
        pred = _logit(model, data, node_feat_mask=m)
        pred_loss = F.binary_cross_entropy_with_logits(pred.view(1), target.view(1))
        size_loss = m.mean()
        ent = -(m * torch.log(m + 1e-15) + (1 - m) * torch.log(1 - m + 1e-15))
        loss = pred_loss + 0.005 * size_loss + 0.1 * ent.mean()
        loss.backward()
        opt.step()
    return torch.sigmoid(mask.detach()).cpu().flatten()


def _ig(model, data, device, steps: int) -> torch.Tensor:
    n = int(data.num_nodes)
    data = data.to(device)
    ones = torch.ones(n, 1, device=device)
    grads = torch.zeros_like(ones)
    for i in range(1, steps + 1):
        m = (ones * (i / steps)).detach().clone().requires_grad_(True)
        pred = _logit(model, data, node_feat_mask=m)
        pred.backward()
        grads = grads + m.grad.detach()
    return (ones * grads / steps).detach().cpu().flatten()


def _occlusion(model, data, device) -> torch.Tensor:
    n = int(data.num_nodes)
    data = data.to(device)
    ones = torch.ones(n, 1, device=device)
    with torch.no_grad():
        base = _logit(model, data).detach()
        imp = torch.empty(n)
        for i in range(n):
            m = ones.clone()
            m[i] = 0
            imp[i] = (base - _logit(model, data, node_feat_mask=m)).detach().cpu()
    return imp


def _cell_types_from_graph(data, rev: dict[int, str]) -> list[str]:
    ids = data.x[:, 0].long().cpu().tolist()
    return [rev.get(int(i), "Unassigned") for i in ids]


def _overlay_catalog_types(
    graph_pkl: Path,
    data,
    region_dir: Path,
    cell_type_col: str,
) -> list[str] | None:
    if not graph_pkl.is_file() or not region_dir.is_dir():
        return None
    types_path = region_dir / "cell_types.csv"
    if not types_path.is_file():
        return None
    import networkx as nx

    graph = pickle.load(graph_pkl.open("rb"))
    table = pd.read_csv(types_path)
    if "cell_id" not in table.columns or cell_type_col not in table.columns:
        return None
    by_id = {
        str(row.cell_id): str(getattr(row, cell_type_col))
        for row in table.itertuples(index=False)
    }
    components = [
        graph.subgraph(nodes)
        for nodes in nx.connected_components(graph)
        if len(nodes) >= len(graph) * 0.1
    ]
    if not components:
        return None
    component_id = int(getattr(data, "component_id", 0))
    if component_id >= len(components):
        return None
    sub = components[component_id]
    ordered = sorted(sub.nodes)
    if len(ordered) != int(data.num_nodes):
        return None
    out = []
    for node in ordered:
        cid = str(sub.nodes[node].get("cell_id"))
        name = by_id.get(cid)
        if name is None:
            return None
        out.append(name)
    return out


def _fold_dirs(cv_dir: Path, seeds, folds) -> list[Path]:
    found = []
    for path in sorted(p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("seed")):
        try:
            seed_s, fold_s, *_ = path.name.split("_")
            seed = int(seed_s.replace("seed", ""))
            fold = int(fold_s.replace("fold", ""))
        except ValueError:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if folds is not None and fold not in folds:
            continue
        if (path / "model_final.pt").is_file():
            found.append(path)
    return found


def _explain_task(args, dataset: str, task: str) -> pd.DataFrame:
    cv_dir = Path(args.ckpt_root) / f"{dataset}_{task}_cv"
    if not cv_dir.is_dir():
        raise FileNotFoundError(f"missing checkpoint dir: {cv_dir}")
    folds = _fold_dirs(cv_dir, args.seeds, args.folds)
    if not folds:
        raise FileNotFoundError(f"no model_final.pt under {cv_dir}")
    if args.max_models is not None:
        folds = folds[: args.max_models]
    print(f"=== {dataset} {task}  models={len(folds)} ===", flush=True)

    vocab = json.loads((folds[0] / "vocab.json").read_text())
    mapping = vocab["cell_type_mapping"]
    rev = _reverse_mapping(mapping)
    graph_dir = cv_dir / "ds" / "graph"
    tg_dir = cv_dir / "ds" / "tg_graph"
    biomarkers = _biomarkers_from_graph(graph_dir)
    node_names = get_feature_names(
        NODE_FEATURES, cell_type_mapping=mapping, biomarkers=biomarkers
    )
    edge_names = get_feature_names(EDGE_FEATURES, cell_type_mapping=mapping, biomarkers=biomarkers)
    mask_fn = _feature_mask(node_names, edge_names)

    ds_cfg = _dataset_cfg(dataset)
    cat_cfg = _catalog_cfg(dataset)
    cell_type_col = cat_cfg.get("cell_type_col", ds_cfg.get("cell_type_col", "cell_type"))
    regions_root = None
    if args.data_root:
        regions_root = Path(args.data_root) / ds_cfg["root"] / ds_cfg.get("regions_dir", "regions")

    gpt_files = sorted(tg_dir.glob("*.gpt"))
    if args.max_regions is not None:
        gpt_files = gpt_files[: args.max_regions]

    sample = torch.load(gpt_files[0], map_location="cpu", weights_only=False)
    num_feat = int(sample.x.shape[1] - 1)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    types: dict[tuple[str, int], str] = {}

    for fold_i, fold_dir in enumerate(folds):
        print(f"  load {fold_dir.name}", flush=True)
        model = _load_model(fold_dir / "model_final.pt", vocab, num_feat, device)
        for gi, gpt in enumerate(gpt_files):
            data = torch.load(gpt, map_location="cpu", weights_only=False)
            rid = str(data.region_id)
            masked = _apply_mask(data, mask_fn)
            if args.method == "gnn-explainer":
                imp = _gnn_explainer(model, masked, device, args.epochs, args.lr)
            elif args.method == "ig":
                imp = _ig(model, masked, device, args.ig_steps)
            elif args.method == "occlusion":
                imp = _occlusion(model, masked, device)
            else:
                raise ValueError(args.method)
            names = _cell_types_from_graph(data, rev)
            if regions_root is not None:
                overlay = _overlay_catalog_types(
                    graph_dir / f"{rid}.gpkl",
                    data,
                    regions_root / rid,
                    cell_type_col,
                )
                if overlay is not None:
                    names = overlay
            imp_list = imp.tolist()
            for node_i, (score, name) in enumerate(zip(imp_list, names)):
                key = (rid, node_i)
                sums[key] = sums.get(key, 0.0) + float(score)
                counts[key] = counts.get(key, 0) + 1
                types[key] = name
            if (gi + 1) % 20 == 0 or gi + 1 == len(gpt_files):
                print(
                    f"    model {fold_i + 1}/{len(folds)}  "
                    f"regions {gi + 1}/{len(gpt_files)}",
                    flush=True,
                )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    rows = [
        dict(
            region_id=rid,
            node_index=node_i,
            cell_type=types[(rid, node_i)],
            importance=sums[(rid, node_i)] / counts[(rid, node_i)],
            task=task,
            method=args.method,
            n_models=counts[(rid, node_i)],
        )
        for rid, node_i in sums
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--method", choices=["gnn-explainer", "ig", "occlusion"], default="gnn-explainer")
    ap.add_argument("--ckpt-root", default=str(_CODE / "model_results" / "SPACE_GM_pseudo"))
    ap.add_argument("--data-root", default=None, help="TME_benchmark_data root; overlays catalog cell types")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--max-regions", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--ig-steps", type=int, default=32)
    args = ap.parse_args()

    tasks = args.tasks or SELECTED.get(args.dataset)
    if not tasks:
        raise ValueError(f"No tasks for {args.dataset}; pass --tasks")

    frames = [_explain_task(args, args.dataset, task) for task in tasks]
    table = pd.concat(frames, ignore_index=True)
    out = Path(
        args.output
        or (
            _CODE
            / "results"
            / "pseudo_label_explanations_panel"
            / "features"
            / f"{args.dataset}_{args.method.replace('-', '_')}.csv"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(table)} rows, {table['region_id'].nunique()} regions)", flush=True)


if __name__ == "__main__":
    main()
