"""Rank unseen films and enrich matches with optional TMDB metadata."""

import json
import re
import unicodedata
from collections import defaultdict
from itertools import islice

import pandas as pd

from src.features import FEATURES, Profile
from src.prepare_data import TMDB, TMDB_FILES


def available_movies(bundle):
    cutoff = pd.to_datetime(bundle["cutoff"], unit="s").normalize()
    movies = bundle["movies"]
    return movies.loc[movies["release_date"].isna() | movies["release_date"].le(cutoff)]


def recommend(bundle, *, user_id=None, ratings=None, top_k=10):
    history = bundle["history"]
    profile = history.users.get(user_id) if user_id is not None else None
    if ratings:
        profile = Profile()
        for movie_id, rating in ratings.items():
            profile.add(movie_id, rating, history.genres[movie_id])
    movies = available_movies(bundle)
    if profile is None:
        counts = movies.index.to_series().map(history.movie_counts).fillna(0).astype(int)
        return movies.assign(rating_count=counts).sort_values("rating_count", ascending=False, kind="stable").head(top_k)
    movies = movies.loc[~movies.index.isin(profile.rated)]
    if movies.empty:
        return movies
    demographics = history.demographics.get(user_id)
    features = pd.DataFrame(
        (history.feature_row(movie_id, profile, demographics) for movie_id in movies.index),
        columns=FEATURES,
    )
    scores = bundle["model"].predict_proba(features)[:, 1]
    return movies.assign(score=scores).sort_values("score", ascending=False, kind="stable").head(top_k)


def _normalize(title):
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    article = re.fullmatch(r"(.+),\s*(The|An|A)", title, flags=re.IGNORECASE)
    if article:
        title = f"{article[2]} {article[1]}"
    text = unicodedata.normalize("NFKD", title.casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(char for char in text
                                                    if not unicodedata.combining(char))))


def load_metadata(movies, directory=TMDB):
    movie_file, credit_file = (directory / name for name in TMDB_FILES)
    if not movie_file.is_file() or not credit_file.is_file():
        return {}, None
    metadata = pd.read_csv(movie_file, usecols=["id", "title", "original_title", "release_date", "overview"])
    metadata["year"] = pd.to_datetime(metadata["release_date"], errors="coerce").dt.year
    credits = pd.read_csv(credit_file, usecols=["movie_id", "cast"]).set_index("movie_id")
    keys = defaultdict(set)
    rows = {}
    for row in metadata.dropna(subset=["year"]).itertuples(index=False):
        rows[row.id] = row
        for title in (row.title, row.original_title):
            if isinstance(title, str):
                keys[(_normalize(title), int(row.year))].add(row.id)
    matches = {}
    ambiguous = 0
    for movie_id, movie in movies.iterrows():
        if pd.isna(movie["release_date"]):
            continue
        ids = keys[(_normalize(movie["title"]), movie["release_date"].year)]
        if len(ids) > 1:
            ambiguous += 1
        if len(ids) != 1:
            continue
        tmdb_id = next(iter(ids))
        cast = json.loads(credits.at[tmdb_id, "cast"]) if tmdb_id in credits.index else []
        overview = rows[tmdb_id].overview
        matches[movie_id] = {
            "tmdb_id": tmdb_id, "overview": overview if isinstance(overview, str) else "",
            "cast": [person["name"] for person in islice(cast, 3)],
        }
    return matches, {"matched": len(matches), "ambiguous": ambiguous, "total": len(movies)}
