"""Build features from ratings available at the prediction time."""

from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter

import numpy as np
import pandas as pd

from src.prepare_data import GENRES

NUMERIC = [
    "year", "movie_mean", "movie_count", "age", "user_mean", "user_count", "genre_similarity",
]
CATEGORICAL = ["gender", "occupation"]
FEATURES = [*GENRES, *NUMERIC, *CATEGORICAL]


@dataclass
class Profile:
    rating_sum: int = 0
    rated: set = field(default_factory=set)
    liked_genres: np.ndarray = field(default_factory=lambda: np.zeros(len(GENRES)))

    def add(self, movie_id, rating, genres):
        self.rating_sum += rating
        self.rated.add(movie_id)
        if rating >= 4:
            self.liked_genres += genres


class History:
    def __init__(self, movies, users):
        self.genres = dict(zip(movies.index, movies.loc[:, list(GENRES)].to_numpy()))
        self.genre_norms = {movie: np.linalg.norm(genres) for movie, genres in self.genres.items()}
        self.years = movies["year"].to_dict()
        self.demographics = users.to_dict(orient="index")
        self.users = {}
        self.movie_sums = {}
        self.movie_counts = {}

    def add_ratings(self, ratings):
        for event in ratings:
            user_id, movie_id, rating = event.user_id, event.movie_id, event.rating
            if user_id not in self.users:
                self.users[user_id] = Profile()
            self.users[user_id].add(movie_id, rating, self.genres[movie_id])
            self.movie_sums[movie_id] = self.movie_sums.get(movie_id, 0) + rating
            self.movie_counts[movie_id] = self.movie_counts.get(movie_id, 0) + 1

    def feature_row(self, movie_id, profile=None, demographics=None):
        genres = self.genres[movie_id]
        count = self.movie_counts.get(movie_id, 0)
        user_count = len(profile.rated) if profile is not None else 0
        similarity = 0.0
        if profile is not None:
            denominator = self.genre_norms[movie_id] * np.linalg.norm(profile.liked_genres)
            if denominator:
                similarity = float(genres @ profile.liked_genres / denominator)
        demographics = demographics or {}
        return (
            *genres, self.years[movie_id], self.movie_sums[movie_id] / count if count else 3.0,
            count, demographics.get("age", np.nan), profile.rating_sum / user_count if user_count else 3.0,
            user_count, similarity, demographics.get("gender", "unknown"),
            demographics.get("occupation", "unknown"),
        )

    def build(self, ratings, *, update=False, include_profiles=True):
        rows = []
        events = ratings.itertuples(index=False)
        if update:
            # Events with the same timestamp cannot see one another.
            for _, events_at_time in groupby(events, key=attrgetter("timestamp")):
                batch = list(events_at_time)
                rows.extend(self._event_features(event) for event in batch)
                self.add_ratings(batch)
        else:
            rows = [self._event_features(event, include_profiles) for event in events]
        return pd.DataFrame(rows, columns=FEATURES, index=ratings.index)

    def _event_features(self, event, include_profiles=True):
        profile = self.users.get(event.user_id) if include_profiles else None
        return self.feature_row(
            event.movie_id, profile, self.demographics.get(event.user_id),
        )


def temporal_split(ratings, seed=42):
    times = ratings["timestamp"]
    train_end, validation_end = times.quantile([0.7, 0.85], interpolation="higher").astype(int)
    user_ids = np.sort(ratings["user_id"].unique())
    cold_ids = np.random.default_rng(seed).choice(user_ids, len(user_ids) // 5, replace=False)
    warm = ~ratings["user_id"].isin(cold_ids)
    train = ratings.loc[warm & (times < train_end)]
    validation = ratings.loc[warm & (times >= train_end) & (times < validation_end)]
    test_period = ratings.loc[times >= validation_end]
    # Keep both withheld users and naturally new users in the no-history cohort.
    has_history = test_period["user_id"].isin(pd.concat([train, validation])["user_id"])
    warm_test = test_period.loc[has_history]
    cold_test = test_period.loc[~has_history]
    withheld_ratings = int(cold_test["user_id"].isin(cold_ids).sum())
    return train, validation, warm_test, cold_test, int(train_end), int(validation_end), withheld_ratings
