"""Version 2 movie features: genre + release year + critic score + director effect.

The Content model only needs 'one numeric vector per movie', so new information is appended after the genre vector.
  G : 18 genre slots (same as v1)
  Y : release year (1985 = 0, unit of 15 years)
  C : critic score (standardized if present, 0 if missing) + a 'no critic score' flag
  D : director effect = how much higher/lower than average the director's 'other movies' were rated
      (the movie's own ratings are excluded = leave-one-movie-out, to prevent answer leakage)
      computed from past history (hist) only; directors with few ratings are shrunk toward 0 (kappa).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import genre_vector
from metadata import build_metadata

FEATURE_SETS = {"G": [], "GY": ["year"], "GYC": ["year", "critic"], "GYCD": ["year", "critic", "director"]}
_META = None


def meta():
    global _META
    if _META is None:
        m = build_metadata()
        _META = m.set_index("movie_id")
    return _META


def director_effects(hist, kappa):
    """Per-director sum and count of (rating - that user's mean). Stored per movie so that 'leave own movie out' is possible."""
    md = meta()
    um = hist.groupby("user_id").rating_norm.transform("mean")
    res = (hist.rating_norm - um).groupby(hist.movie_id).agg(["sum", "count"])
    d_sum, d_cnt, per_movie = defaultdict(float), defaultdict(float), {}
    for m, (s, c) in res.iterrows():
        dirs = md.directors.get(m) if m in md.index else None
        if isinstance(dirs, list):
            per_movie[m] = (dirs, s, c)
            for d in dirs:
                d_sum[d] += s
                d_cnt[d] += c
    out = {}
    for m in md.index:
        dirs = md.directors.get(m)
        if not isinstance(dirs, list):
            continue
        own = per_movie.get(m)
        vals = []
        for d in dirs:
            s, c = d_sum.get(d, 0.0), d_cnt.get(d, 0.0)
            if own:
                s, c = s - own[1], c - own[2]
            vals.append(s / (c + kappa))
        out[m] = float(np.mean(vals))
    return out


def extra_vector(m, parts, deff):
    md = meta()
    v = []
    known = m in md.index
    if "year" in parts:
        v.append((md.year.get(m) - 1985) / 15.0 if known else 0.0)
    if "critic" in parts:
        c = md.critic.get(m) if known else np.nan
        v += [0.0, 1.0] if pd.isna(c) else [(c - 0.6) / 0.25, 0.0]
    if "director" in parts:
        v.append(deff.get(m, 0.0))
    return np.array(v, float)


def build_table(hist, genre_table, feature_set, kappa=50.0, extra_ids=()):
    """Table of movie_id -> [genre | extra features] vectors. hist contains only ratings 'before the prediction time'."""
    parts = FEATURE_SETS[feature_set]
    deff = director_effects(hist, kappa) if "director" in parts else {}
    ids = set(genre_table) | set(meta().index) | set(extra_ids)
    zero_g = np.zeros(18)
    return {m: np.concatenate([genre_table.get(m, zero_g), extra_vector(m, parts, deff)]) for m in ids}


def extend_with_genres(table_v2, other_genre_table, feature_set):
    """Use the genre notation of another dataset (MovieTweetings/Beliefs), but take the extra features from MovieLens when a MovieLens id exists."""
    n_extra = len(next(iter(table_v2.values()))) - 18
    out = {}
    for m, g in other_genre_table.items():
        base = table_v2.get(m)
        extra = base[18:] if base is not None else _missing_extra(feature_set)
        out[m] = np.concatenate([g, extra]) if n_extra else g
    return out


def _missing_extra(feature_set):
    parts = FEATURE_SETS[feature_set]
    v = []
    if "year" in parts:
        v.append(0.0)
    if "critic" in parts:
        v += [0.0, 1.0]
    if "director" in parts:
        v.append(0.0)
    return np.array(v, float)


def risk_item_table(hist, kappa=50.0):
    """Movie information for the risk model: critic score (with a missing flag), release year, director effect."""
    md = meta()
    deff = director_effects(hist, kappa)
    df = pd.DataFrame({"critic": md.critic, "year": md.year})
    df["critic_missing"] = df.critic.isna().astype(float)
    df["critic"] = df.critic.fillna(df.critic.mean())
    df["year_z"] = (df.year - 1985) / 15.0
    df["dir_effect"] = pd.Series(deff).reindex(df.index).fillna(0.0)
    return df[["critic", "critic_missing", "year_z", "dir_effect"]]


def user_genre_dislike(hist, thr, alpha):
    """Per-user 'Strong Dislike rate by genre' (18 slots). Genres with little history are shrunk toward the person's overall Strong Dislike rate (alpha)."""
    h = hist[["user_id", "movie_id", "rating_norm", "genres"]].copy()
    h["dis"] = (h.rating_norm <= h.user_id.map(thr)).astype(float)
    G = np.vstack([(genre_vector(g) > 0).astype(float) for g in h.genres])
    out = {}
    codes, uniq = pd.factorize(h.user_id)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    dis = h.dis.to_numpy()
    for j, u in enumerate(uniq):
        rows = order[bounds[j]:bounds[j + 1]]
        base = dis[rows].mean()
        n = G[rows].sum(0)
        k = (G[rows] * dis[rows, None]).sum(0)
        out[u] = (k + alpha * base) / (n + alpha)
    return out
