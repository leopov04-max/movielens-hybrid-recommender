"""Evaluate observed ratings and diagnose errors by movie popularity."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)


def choose_threshold(labels, scores):
    thresholds = np.linspace(0.1, 0.9, 33)
    return float(max(thresholds, key=lambda t: (
        f1_score(labels, scores >= t, zero_division=0),
        precision_score(labels, scores >= t, zero_division=0),
    )))


def ranking_metrics(events, scores, cutoffs=(5, 10)):
    ranked = events[["user_id", "movie_id", "liked"]].assign(score=scores)
    ranked = ranked.sort_values(["user_id", "score", "movie_id"], ascending=[True, False, True])
    groups = ranked.groupby("user_id", sort=False)
    sizes = groups["liked"].transform("size")
    order = groups.cumcount()
    result = {}
    for k in cutoffs:
        eligible = ranked.loc[(sizes >= k) & (order < k)]
        per_user = eligible.groupby("user_id")["liked"].mean()
        result[f"precision_at_{k}"] = float(per_user.mean()) if len(per_user) else None
        result[f"users_at_{k}"] = len(per_user)
    return result


def evaluate(events, scores, threshold):
    labels = events["liked"].to_numpy()
    predicted = scores >= threshold
    return {
        "ratings": len(events),
        "users": int(events["user_id"].nunique()),
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        **ranking_metrics(events, scores),
    }


def popularity_errors(events, scores, threshold, prior_counts):
    low, high = np.quantile(list(prior_counts.values()), [0.25, 0.75])
    counts = events["movie_id"].map(prior_counts).fillna(0)
    cohorts = {
        "unseen": counts == 0,
        "long_tail": (counts > 0) & (counts <= low),
        "middle": (counts > low) & (counts <= high),
        "popular": counts > high,
    }
    rows = []
    for name, mask in cohorts.items():
        actual = events.loc[mask, "liked"]
        predicted = scores[mask.to_numpy()] >= threshold
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        rows.append({
            "group": name, "ratings": len(actual), "false_positives": int(fp),
            "false_negatives": int(fn),
            "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        })
    return {"p25_count": float(low), "p75_count": float(high), "groups": rows}
