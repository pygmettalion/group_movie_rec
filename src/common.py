"""Shared tools: data loading, genre vectors, scoring functions.

In plain words: the 'shared drawer' that every stage uses.
- All ratings use rating_norm, rescaled to 0~1 (each dataset has a different star range).
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))  # version 2 uses RESULTS_DIR=results_v2
RESULTS.mkdir(exist_ok=True)

GENRES = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']
G_INDEX = {g: i for i, g in enumerate(GENRES)}

SEED = 42


def read_csv(rel):
    return pd.read_csv(DATA / rel, encoding="utf-8-sig")


def load_movielens():
    r = read_csv("01_MovieLens_1M/ratings.csv")
    groups = read_csv("01_MovieLens_1M/eval_groups.csv")
    return r, groups


def load_movietweetings():
    r = read_csv("02_MovieTweetings/movietweetings_ratings.csv")
    r["movie_id"] = r["movie_id"].astype(str)
    groups = read_csv("02_MovieTweetings/movietweetings_test_groups.csv")
    return r, groups


def load_beliefs():
    test = read_csv("03_Beliefs/unseen_test.csv")
    hist = read_csv("03_Beliefs/past_history.csv")
    return test, hist


def genre_vector(genres):
    """Genre string 'Action|Comedy' -> an 18-slot vector, divided so that it sums to 1 (genre 'composition')."""
    v = np.zeros(len(GENRES))
    if isinstance(genres, str):
        for g in genres.split("|"):
            if g in G_INDEX:
                v[G_INDEX[g]] = 1.0
    s = v.sum()
    return v / s if s > 0 else v


def movie_genre_table(*frames):
    """Build a movie_id -> genre vector dictionary from several rating tables."""
    table = {}
    for f in frames:
        for mid, g in f[["movie_id", "genres"]].drop_duplicates("movie_id").itertuples(index=False):
            if mid not in table or (isinstance(g, str) and table[mid].sum() == 0):
                table[mid] = genre_vector(g)
    return table


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def save_json(obj, name):
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def group_rows(groups, split):
    """Extract a list of (group_id, [4 users], [candidate movies]) from a group table."""
    out = []
    col = f"{split}_movie_ids"
    for row in groups.itertuples(index=False):
        movies = getattr(row, col)
        if not isinstance(movies, str) or not movies:
            continue
        users = [getattr(row, f"user_id_{i}") for i in range(1, 5)]
        out.append((row.group_id, users, [int(m) for m in movies.split("|")]))
    return out
