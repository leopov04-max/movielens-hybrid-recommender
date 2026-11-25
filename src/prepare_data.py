"""Download the datasets and load the fields used by the project."""

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MOVIELENS = RAW / "ml-100k"
TMDB = RAW / "tmdb"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports" / "generated"
MODEL_NAME = "recommender.joblib"
GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_CHECKSUM = "0e33842e24a9c977be4e0107933c0723"
TMDB_URL = "https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata"
TMDB_FILES = ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")
DOWNLOAD_HEADERS = {"Accept": "application/zip", "User-Agent": "movielens-hybrid-recommender/1.0"}


def download(url, destination, members, *, checksum=None):
    missing = {name: member for name, member in members.items()
               if not (destination / name).is_file()}
    if not missing:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as temp:
        archive_path = Path(temp) / "dataset.zip"
        with requests.get(url, headers=DOWNLOAD_HEADERS, timeout=(10, 60)) as response:
            response.raise_for_status()
            archive_path.write_bytes(response.content)
        if checksum:
            with archive_path.open("rb") as downloaded:
                digest = hashlib.file_digest(downloaded, lambda: hashlib.md5(usedforsecurity=False))
            if digest.hexdigest() != checksum:
                raise ValueError("MovieLens archive does not match the GroupLens checksum.")
        with ZipFile(archive_path) as archive:
            for name, member in missing.items():
                temporary_file = Path(temp) / name
                with archive.open(member) as source, temporary_file.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary_file.replace(destination / name)


def load_movielens(directory=MOVIELENS):
    ratings = pd.read_csv(
        directory / "u.data", sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    ).sort_values("timestamp", kind="stable", ignore_index=True)
    users = pd.read_csv(
        directory / "u.user", sep="|", encoding="latin-1",
        names=["user_id", "age", "gender", "occupation", "zip"],
    ).drop(columns="zip").set_index("user_id")
    movies = pd.read_csv(
        directory / "u.item", sep="|", encoding="latin-1",
        names=["movie_id", "title", "release_date", "video_date", "imdb_url", *GENRES],
    ).drop(columns=["video_date", "imdb_url"]).set_index("movie_id")
    movies["release_date"] = pd.to_datetime(movies["release_date"], format="%d-%b-%Y")
    movies["year"] = movies["release_date"].dt.year
    ratings["liked"] = (ratings["rating"] >= 4).astype(int)
    return ratings, users, movies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmdb", action="store_true", help="Also download optional TMDB metadata.")
    args = parser.parse_args()
    download(
        ML_URL, MOVIELENS, {name: f"ml-100k/{name}" for name in ("u.data", "u.user", "u.item")},
        checksum=ML_CHECKSUM,
    )
    print(f"MovieLens: {MOVIELENS}")
    if args.tmdb:
        download(TMDB_URL, TMDB, {name: name for name in TMDB_FILES})
        print(f"TMDB: {TMDB}")


if __name__ == "__main__":
    main()
