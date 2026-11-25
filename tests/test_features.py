import unittest

import pandas as pd

from src.features import History
from src.prepare_data import GENRES


class HistoryTests(unittest.TestCase):
    def test_features_only_include_strictly_earlier_events(self):
        movies = pd.DataFrame(0, index=[1, 2, 3], columns=GENRES)
        movies["year"] = 1990
        movies["Action"] = 1
        users = pd.DataFrame({"age": [30], "gender": ["F"], "occupation": ["student"]}, index=[1])
        ratings = pd.DataFrame([
            (1, 1, 5, 1), (1, 2, 1, 1), (1, 3, 4, 2), (2, 1, 2, 3),
        ], columns=["user_id", "movie_id", "rating", "timestamp"])

        features = History(movies, users).build(ratings, update=True)

        self.assertEqual(features["user_count"].tolist(), [0, 0, 2, 0])
        self.assertEqual(features["user_mean"].tolist(), [3, 3, 3, 3])
        self.assertEqual(features["movie_count"].tolist(), [0, 0, 0, 1])
        self.assertEqual(features["movie_mean"].tolist(), [3, 3, 3, 5])
        self.assertEqual(features["genre_similarity"].tolist(), [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
