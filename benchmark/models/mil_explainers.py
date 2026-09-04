"""Instance explainers and perturbation faithfulness for gated AttnMIL.

Attention is the baseline, not the decision decomposition.  Single and
one-removed follow MILLI; IG attributes the positive logit to window features
and sums per instance.  Random is the null ranking for AUPC tests.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from benchmark.models.attention_mil import AttentionMILModel

EXPLAINERS = ("attention", "single", "one_removed", "ig", "random")


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


def ig_scores(model: AttentionMILModel, bag: np.ndarray, steps: int = 32) -> np.ndarray:
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
    return attr.sum(dim=1).detach().cpu().numpy().astype(float)


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
