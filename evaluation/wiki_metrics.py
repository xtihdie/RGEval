from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .wiki_quality_support import WIKI_LABEL_TO_SCORE


def _normalize_labels(values: Iterable[object]) -> list[float]:
    scores: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        scores.append(float(WIKI_LABEL_TO_SCORE[str(value).strip()]))
    return scores


def _quadratic_weighted_kappa(y_true: list[float], y_pred: list[float]) -> float:
    categories = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    index = {value: i for i, value in enumerate(categories)}
    n = len(categories)
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    for truth, pred in zip(y_true, y_pred):
        observed[index[truth]][index[pred]] += 1.0

    true_hist = [0.0 for _ in range(n)]
    pred_hist = [0.0 for _ in range(n)]
    for truth in y_true:
        true_hist[index[truth]] += 1.0
    for pred in y_pred:
        pred_hist[index[pred]] += 1.0

    total = float(len(y_true))
    expected = [[(true_hist[i] * pred_hist[j]) / total for j in range(n)] for i in range(n)]

    weight_sum_observed = 0.0
    weight_sum_expected = 0.0
    denom = float((n - 1) ** 2) if n > 1 else 1.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / denom
            weight_sum_observed += weight * observed[i][j]
            weight_sum_expected += weight * expected[i][j]

    if weight_sum_expected == 0:
        return 1.0
    return 1.0 - (weight_sum_observed / weight_sum_expected)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def compute_label_metrics(df: pd.DataFrame, *, truth_col: str, pred_col: str) -> dict[str, float]:
    subset = df[[truth_col, pred_col]].dropna().copy()
    if subset.empty:
        return {
            "count": 0,
            "qwk": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "exact_accuracy": float("nan"),
            "within_0_5_accuracy": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    y_true = _normalize_labels(subset[truth_col])
    y_pred = _normalize_labels(subset[pred_col])
    abs_errors = [abs(a - b) for a, b in zip(y_true, y_pred)]
    sq_errors = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    exact = [1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)]
    within = [1.0 if abs(a - b) <= 1.0 else 0.0 for a, b in zip(y_true, y_pred)]

    return {
        "count": len(y_true),
        "qwk": _quadratic_weighted_kappa(y_true, y_pred),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)),
        "exact_accuracy": sum(exact) / len(exact),
        "within_0_5_accuracy": sum(within) / len(within),
        "pearson": _pearson(y_true, y_pred),
        "spearman": _pearson(_rank(y_true), _rank(y_pred)),
    }
