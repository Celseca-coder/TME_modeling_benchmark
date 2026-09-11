"""Instance explainers and perturbation faithfulness for gated AttnMIL.

Attention is the baseline, not the decision decomposition.  Single and
one-removed follow MILLI; IG attributes the positive logit to window features
and sums per instance.  Random is the null ranking for AUPC tests.

Attention, single and one-removed are structurally instance-level: they emit one
number per window and cannot say which measurement inside that window mattered.
IG is the exception — it produces a full ``(window, feature)`` matrix, so
:func:`ig_attributions` keeps that matrix and :func:`aggregate_feature_attributions`
collapses it onto named window features for a Lasso/SHAP-style ranking.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from benchmark.models.attention_mil import AttentionMILModel

EXPLAINERS = ("attention", "single", "one_removed", "ig", "random")
FEATURE_AGGS = ("abs", "signed", "top")


def _as_bag(bag: np.ndarray) -> np.ndarray:
    bag = np.asarray(bag, dtype=np.float32)
    if bag.ndim != 2:
        raise ValueError(f"bag must be 2-D, got {bag.shape}")
    return bag


def attention_scores(model: AttentionMILModel, bag: np.ndarray) -> np.ndarray:
    model._net.eval()
    with torch.no_grad():
        _, weights = model._net(model._bag_tensor(_as_bag(bag)))
    return weights.cpu().numpy().astype(float)


def single_scores(model: AttentionMILModel, bag: np.ndarray) -> np.ndarray:
    bag = _as_bag(bag)
    scores = np.zeros(len(bag), dtype=float)
    for i in range(len(bag)):
        scores[i] = model.positive_logit(bag[i : i + 1])
    return scores


def one_removed_scores(model: AttentionMILModel, bag: np.ndarray) -> np.ndarray:
    bag = _as_bag(bag)
    n = len(bag)
    scores = np.zeros(n, dtype=float)
    if n == 1:
        scores[0] = model.positive_logit(bag)
        return scores
    full = model.positive_logit(bag)
    for i in range(n):
        remain = np.delete(bag, i, axis=0)
        scores[i] = full - model.positive_logit(remain)
    return scores


def ig_attributions(model: AttentionMILModel, bag: np.ndarray, steps: int = 32) -> np.ndarray:
    """Integrated Gradients of the positive logit, shaped ``(window, feature)``.

    The path baseline is the zero vector of the model's scaled input space, which
    is the mean training window, so an entry reads as "how much this window's
    deviation from an average window pushed the positive logit".  Attributions
    therefore live in scaled units and are comparable across features.
    """
    bag = _as_bag(bag)
    model._net.eval()
    x = model._bag_tensor(bag)
    baseline = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    pos = model._positive_index()
    for i in range(1, int(steps) + 1):
        model._net.zero_grad(set_to_none=True)
        t = (baseline + (i / steps) * (x - baseline)).detach().requires_grad_(True)
        logit = model.forward_scaled(t)[0][pos]
        logit.backward()
        acc = acc + t.grad.detach()
    attr = (x - baseline) * acc / steps
    return attr.detach().cpu().numpy().astype(float)


def ig_scores(model: AttentionMILModel, bag: np.ndarray, steps: int = 32) -> np.ndarray:
    return ig_attributions(model, bag, steps=steps).sum(axis=1)


def aggregate_feature_attributions(
    attr: np.ndarray,
    mode: str = "abs",
    *,
    instance_scores: np.ndarray | None = None,
    top_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a ``(window, feature)`` attribution matrix onto features.

    Returns ``(importance, net)``.  ``importance`` is non-negative and is what
    the ranking uses; ``net`` keeps the sign, so a reader can tell whether the
    feature pushed toward the positive class or away from it.

    ``abs``     mean absolute attribution over every window; the direct
                analogue of the mean-|SHAP| ranking used on global features.
    ``signed``  attribution summed over windows, ranked by magnitude, so
                windows carrying opposite evidence cancel.
    ``top``     mean absolute attribution over the top ``top_frac`` windows
                only, i.e. "inside the windows the model actually used, which
                measurement drove it".
    """
    attr = np.asarray(attr, dtype=float)
    if attr.ndim != 2:
        raise ValueError(f"attributions must be 2-D, got {attr.shape}")
    if attr.size == 0:
        empty = np.zeros(attr.shape[1], dtype=float)
        return empty, empty.copy()
    if mode == "signed":
        net = attr.sum(axis=0)
        return np.abs(net), net
    if mode == "top":
        scores = attr.sum(axis=1) if instance_scores is None else np.asarray(instance_scores, dtype=float)
        k = max(1, int(np.ceil(float(top_frac) * len(attr))))
        attr = attr[np.argsort(-scores)[:k]]
    elif mode != "abs":
        raise ValueError(f"Unknown feature aggregation {mode!r}; expected {FEATURE_AGGS}")
    return np.abs(attr).mean(axis=0), attr.mean(axis=0)


def rank_feature_order(importance: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Indices of ``feature_names`` sorted by descending importance."""
    importance = np.asarray(importance, dtype=float)
    if importance.shape[0] != len(feature_names):
        raise ValueError(
            f"{importance.shape[0]} importances for {len(feature_names)} feature names"
        )
    return np.argsort(-np.nan_to_num(importance, nan=-np.inf))


def random_scores(bag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(len(_as_bag(bag)))


SCORERS: dict[str, Callable] = {
    "attention": attention_scores,
    "single": single_scores,
    "one_removed": one_removed_scores,
    "ig": ig_scores,
}


def explain_bag(
    model: AttentionMILModel,
    bag: np.ndarray,
    explainer: str,
    rng: np.random.Generator | None = None,
    ig_steps: int = 32,
) -> np.ndarray:
    name = str(explainer)
    if name == "random":
        return random_scores(bag, rng or np.random.default_rng(0))
    if name == "ig":
        return ig_scores(model, bag, steps=ig_steps)
    if name not in SCORERS:
        raise ValueError(f"Unknown MIL explainer {name!r}; expected {EXPLAINERS}")
    return SCORERS[name](model, bag)


def _proba_or_neutral(model: AttentionMILModel, bag: np.ndarray) -> float:
    if bag is None or len(bag) == 0:
        return 0.5
    return model.positive_proba(bag)


def perturbation_curve(
    model: AttentionMILModel,
    bag: np.ndarray,
    scores: np.ndarray,
    *,
    mode: str,
    step_frac: float = 0.10,
) -> np.ndarray:
    """MORF / LERF / insertion curves of positive-class probability.

    ``mode``:
      morf — drop highest scores first
      lerf — drop lowest scores first
      insertion — start empty, add highest first
    """
    bag = _as_bag(bag)
    n = len(bag)
    if n == 0:
        return np.asarray([0.5], dtype=float)
    step = max(1, int(np.ceil(step_frac * n)))
    order = np.argsort(-np.asarray(scores, dtype=float))
    points = []
    if mode == "insertion":
        kept: list[int] = []
        points.append(_proba_or_neutral(model, bag[0:0]))
        for start in range(0, n, step):
            kept.extend(int(i) for i in order[start : start + step])
            points.append(_proba_or_neutral(model, bag[np.asarray(kept, dtype=int)]))
        return np.asarray(points, dtype=float)
    if mode == "lerf":
        drop_order = order[::-1]
    elif mode == "morf":
        drop_order = order
    else:
        raise ValueError(f"Unknown perturbation mode {mode!r}")
    remaining = np.arange(n)
    points.append(_proba_or_neutral(model, bag))
    for start in range(0, n, step):
        drop = set(int(i) for i in drop_order[start : start + step])
        remaining = np.asarray([i for i in remaining if i not in drop], dtype=int)
        points.append(_proba_or_neutral(model, bag[remaining] if len(remaining) else bag[0:0]))
    return np.asarray(points, dtype=float)


def aupc(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    if curve.size == 1:
        return float(curve[0])
    x = np.linspace(0.0, 1.0, num=len(curve))
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(curve, x))


def faithfulness_metrics(
    model: AttentionMILModel,
    bag: np.ndarray,
    scores: np.ndarray,
    random_scores_arr: np.ndarray,
    step_frac: float = 0.10,
) -> dict[str, float]:
    morf = perturbation_curve(model, bag, scores, mode="morf", step_frac=step_frac)
    lerf = perturbation_curve(model, bag, scores, mode="lerf", step_frac=step_frac)
    insert = perturbation_curve(model, bag, scores, mode="insertion", step_frac=step_frac)
    morf_rand = perturbation_curve(model, bag, random_scores_arr, mode="morf", step_frac=step_frac)
    aupc_morf = aupc(morf)
    aupc_lerf = aupc(lerf)
    aupc_rand = aupc(morf_rand)
    return dict(
        aupc_morf=aupc_morf,
        aupc_lerf=aupc_lerf,
        aupc_insert=aupc(insert),
        aupc_random=aupc_rand,
        delta_lerf_morf=aupc_lerf - aupc_morf,
        faith_passed=bool(
            np.isfinite(aupc_morf)
            and aupc_morf < aupc_rand
            and (aupc_lerf - aupc_morf) > 0
        ),
    )
