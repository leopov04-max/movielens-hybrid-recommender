"""Train and compare KNN, Logistic Regression and CART on MovieLens 100K."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import matplotlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

from src.evaluate import choose_threshold, evaluate, popularity_errors, ranking_metrics
from src.features import CATEGORICAL, NUMERIC, History, temporal_split
from src.prepare_data import ARTIFACTS, GENRES, MODEL_NAME, REPORTS, load_movielens

matplotlib.use("Agg")
from matplotlib import pyplot as plt

def candidates(seed):
    return {
        "Logistic Regression": [({"C": c}, LogisticRegression(C=c, max_iter=500))
                                for c in (0.1, 1.0, 10.0)],
        "KNN": [({"n_neighbors": k}, KNeighborsClassifier(n_neighbors=k, algorithm="brute"))
                for k in (15, 35, 75)],
        "CART": [({"max_depth": depth}, DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=30, random_state=seed,
        )) for depth in (4, 8, 12)],
    }


def pipeline(estimator):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("genres", "passthrough", list(GENRES)),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def validation_rank(metrics):
    at_five = metrics["precision_at_5"]
    return (at_five if at_five is not None else -1.0, metrics["f1"], metrics["precision"])


def compare(train, validation, warm, cold, features, seed):
    records = {}
    models = {}
    test_scores = {}
    with threadpool_limits(limits=4):
        for name, options in candidates(seed).items():
            print(f"Training {name}...", flush=True)
            evaluated = []
            for params, estimator in options:
                fitted = pipeline(estimator).fit(features["train"], train["liked"])
                scores = fitted.predict_proba(features["validation"])[:, 1]
                threshold = choose_threshold(validation["liked"], scores)
                metrics = evaluate(validation, scores, threshold)
                evaluated.append((metrics, params, threshold, fitted))
            metrics, params, threshold, fitted = max(evaluated, key=lambda opt: validation_rank(opt[0]))
            models[name] = fitted
            test_scores[name] = fitted.predict_proba(features["warm"])[:, 1]
            records[name] = {
                "params": params, "threshold": threshold, "validation": metrics,
                "tried": [{"params": settings, "threshold": cut, "metrics": report}
                          for report, settings, cut, _ in evaluated],
                "warm_test": evaluate(warm, test_scores[name], threshold),
                "cold_test": evaluate(cold, fitted.predict_proba(features["cold"])[:, 1], threshold),
            }
    return records, models, test_scores


def save_plot(report, destination):
    models = report["models"]
    metrics = ("precision", "precision_at_5", "precision_at_10")
    names = list(models) + ["Popularity"]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, metric, color in zip((-0.24, 0, 0.24), metrics, ("#146c72", "#4873ad", "#bd6736")):
        values = [models[name]["warm_test"][metric] for name in models]
        values.append(report["popularity_baseline"]["warm_test"].get(metric))
        ax.bar(positions + offset, [np.nan if value is None else value for value in values],
               width=0.23, label=metric.replace("_at_", "@"), color=color)
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 1), xlabel="Model", ylabel="Score", title="MovieLens 100K · Warm test")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_artifact(bundle, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def explain_models(models):
    logistic = models["Logistic Regression"]
    names = logistic["preprocess"].get_feature_names_out()
    weights = logistic["classifier"].coef_[0]
    order = np.argsort(np.abs(weights))[::-1][:10]
    tree = models["CART"]
    return {
        "logistic_largest_coefficients": {names[i]: float(weights[i]) for i in order},
        "cart_first_levels": export_text(
            tree["classifier"],
            feature_names=list(tree["preprocess"].get_feature_names_out()),
            max_depth=3,
        ),
    }


def error_examples(events, scores, threshold, counts, bands, titles):
    counts = events["movie_id"].map(counts).fillna(0)
    rows = events[["movie_id", "rating"]].assign(score=scores, prior_movie_ratings=counts)
    masks = {
        "popular_false_positive": (counts > bands["p75_count"]) & (events["liked"] == 0)
        & (scores >= threshold),
        "long_tail_false_negative": (counts > 0) & (counts <= bands["p25_count"])
        & (events["liked"] == 1) & (scores < threshold),
    }
    result = {}
    for name, mask in masks.items():
        examples = rows.loc[mask].sort_values("score", ascending=name.endswith("negative")).head(3)
        examples = examples.assign(title=examples["movie_id"].map(titles)).drop(columns="movie_id")
        result[name] = examples.to_dict(orient="records")
    return result


def run(artifact_dir, report_dir):
    seed = 42
    ratings, users, movies = load_movielens()
    train, validation, warm, cold, train_end, validation_end, withheld_ratings = temporal_split(ratings, seed)
    history = History(movies, users)
    features = {"train": history.build(train, update=True), "validation": history.build(validation)}
    history.add_ratings(validation.itertuples(index=False))
    features["warm"] = history.build(warm)
    features["cold"] = history.build(cold)
    records, models, test_scores = compare(train, validation, warm, cold, features, seed)
    winner = max(records, key=lambda name: validation_rank(records[name]["validation"]))
    threshold = records[winner]["threshold"]
    bands = popularity_errors(warm, test_scores[winner], threshold, history.movie_counts)
    with threadpool_limits(limits=4):
        scores_without_history = models[winner].predict_proba(history.build(warm, include_profiles=False))[:, 1]
    report = {
        "seed": seed,
        "dataset": {"ratings": len(ratings), "movies": len(movies), "users": len(users),
                    "positive_fraction": float(ratings["liked"].mean())},
        "split": {
            "validation_start_utc": datetime.fromtimestamp(train_end, timezone.utc).isoformat(),
            "test_start_utc": datetime.fromtimestamp(validation_end, timezone.utc).isoformat(),
            "train": len(train), "validation": len(validation), "warm_test": len(warm),
            "cold_test": len(cold),
            "cold_test_withheld_ratings": withheld_ratings,
            "cold_test_naturally_new_ratings": len(cold) - withheld_ratings,
        },
        "selected_model": winner,
        "warm_without_user_history": evaluate(warm, scores_without_history, threshold),
        "models": records,
        "popularity_baseline": {
            "warm_test": ranking_metrics(warm, features["warm"]["movie_count"]),
            "cold_test": ranking_metrics(cold, features["cold"]["movie_count"]),
        },
        "popularity_errors": bands,
        "error_examples": error_examples(
            warm, test_scores[winner], threshold, history.movie_counts, bands, movies["title"],
        ),
        "interpretability": explain_models(models),
    }
    save_artifact(
        {"history": history, "movies": movies[["title", "release_date", *GENRES]],
         "model": models[winner], "model_name": winner, "cutoff": validation_end},
        artifact_dir / MODEL_NAME,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    save_plot(report, report_dir / "comparison.png")
    print(f"Selected model: {winner}\nModel: {artifact_dir / MODEL_NAME}\nMetrics: {report_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    args = parser.parse_args()
    run(args.artifact_dir, args.report_dir)


if __name__ == "__main__":
    main()
