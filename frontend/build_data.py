"""Build the data file for the demo page (frontend/index.html) with the frozen v1 model.

Usage (from the repository root):  python frontend/build_data.py
Output: frontend/data.js  (the page reads it directly, so it opens without a server)

The model is not changed: this script only calls the frozen v1 model (results_v1_frozen/) with exactly
the same procedure as 05_demo.py, for each of the 49 official groups plus 200 random groups of 4 users, and saves what the page shows.
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("RESULTS_DIR", str(ROOT / "results_v1_frozen"))
sys.path.insert(0, str(ROOT / "src"))
from common import RESULTS, SEED, load_json, load_movielens, movie_genre_table  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import indiv_features, movie_stats, user_thresholds  # noqa: E402

TOP_N = 5


def stars(x):
    """0~1 normalized score -> 1~5 stars."""
    return float(1 + 4 * np.clip(x, 0, 1))


final = load_json("FINAL_FROZEN_CONFIG.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
r, groups = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
gt = movie_genre_table(r)
s1 = Stage1Predictor(final["stage1"], gt, seed=SEED).fit(trva)
th = user_thresholds(trva, final["stage2"]["percentile"])
mstats = movie_stats(trva)
lam = float(final["lam"])

# same reliability filter as 05_demo.py (cutoff computed from the Validation error by training-rating-count bins)
tr = r[r.split == "train"]
v = pd.read_csv(RESULTS / "val_stage1_predictions.csv")
v["n"] = v.movie_id.map(tr.groupby("movie_id").size())
v["bucket"] = pd.cut(v.n, [0, 5, 10, 20, 50, 100, 200, 500, 10**9])
by = v.groupby("bucket", observed=True).apply(lambda d: np.sqrt(((d.rating_norm - d.pred) ** 2).mean()))
overall = np.sqrt(((v.rating_norm - v.pred) ** 2).mean())
bad = [b for b, e in by.items() if e > 1.1 * overall]
min_count = int(max(b.right for b in bad)) + 1 if bad else 1
n_trva = trva.groupby("movie_id").size()
all_movies = sorted(set(trva.movie_id))

genre_str = r.drop_duplicates("movie_id").set_index("movie_id").genres
titles = {}
movies_dat = ROOT / "data_extra" / "ml-1m" / "movies.dat"
if movies_dat.exists():
    mv = pd.read_csv(movies_dat, sep="::", engine="python", names=["movie_id", "title", "g"], encoding="latin1")
    titles = dict(zip(mv.movie_id, mv.title))


def split_title(m):
    """'Sanjuro (1962)' -> ('Sanjuro', 1962). Moves a trailing article to the front: 'Celebration, The (Festen)' -> 'The Celebration (Festen)'."""
    t = titles.get(m, f"Movie #{m}")
    year = None
    mm = re.match(r"^(.*) \((\d{4})\)$", t)
    if mm:
        t, year = mm.group(1), int(mm.group(2))
    mm = re.match(r"^(.*), (The|A|An)( \(.*\))?$", t)
    if mm:
        t = f"{mm.group(2)} {mm.group(1)}{mm.group(3) or ''}"
    return t, year


HIGH = trva[trva.rating_norm >= 0.75]
GENRE_BASE = HIGH.genres.str.split("|").explode().value_counts(normalize=True)


def favorite_genres(u, k=2):
    """Genres this user rates 4-5 stars more often than everyone else does (lift over the overall share), for the seat card."""
    g = HIGH[HIGH.user_id == u].genres.str.split("|").explode()
    cnt = g.value_counts()
    lift = (cnt / cnt.sum() / GENRE_BASE.reindex(cnt.index))[cnt >= 5]
    return lift.sort_values(ascending=False).index[:k].tolist()


def movie_entry(i, cand, pred, pi, E, R, F):
    m = cand[i]
    title, year = split_title(m)
    return dict(id=int(m), title=title, year=year, genres=genre_str[m].split("|"),
                exp=stars(E[i]), risk=float(R[i]), score=float(F[i]),
                members=[dict(pred=stars(pred[i, j]), p=float(pi[i, j])) for j in range(4)])


def build_group(gid, users, kind):
    seen = set(r[r.user_id.isin(users)].movie_id)
    cand = [m for m in all_movies if m not in seen and s1.mf_knows_item([m])[0] and gt[m].sum() > 0
            and n_trva.get(m, 0) >= min_count]
    pred = np.column_stack([s1.predict([u] * len(cand), cand) for u in users])
    thr = th.thr.loc[users].to_numpy()
    share = th.share.loc[users].to_numpy()
    # member-level Strong Dislike probabilities p_i (the same computation as group.risk_proba, kept per member for display)
    pi = np.column_stack([
        risk["model"].predict_proba(indiv_features(pred[:, j], np.full(len(cand), thr[j]), np.full(len(cand), share[j]),
                                                   np.array(cand), mstats)[risk["features"]])[:, 1]
        for j in range(4)])
    R = 1 - np.prod(1 - pi, axis=1)
    E = pred.mean(1)
    F = E - lam * R
    order = np.argsort(-F, kind="stable")
    avg_i = int(np.argmax(E))
    return dict(
        id=gid, kind=kind, candidates=len(cand),
        members=[dict(user=u, threshold=stars(thr[j]), ratings=int((trva.user_id == u).sum()),
                      likes=favorite_genres(u)) for j, u in enumerate(users)],
        picks=[movie_entry(int(i), cand, pred, pi, E, R, F) for i in order[:TOP_N]],
        average=movie_entry(avg_i, cand, pred, pi, E, R, F))


out = []
# 1) the 49 official evaluation groups
for _, row in groups.iterrows():
    out.append(build_group(int(row.group_id), [int(row[f"user_id_{i}"]) for i in range(1, 5)], "official"))
    print(f"official group {int(row.group_id):2d} -> {out[-1]['picks'][0]['title']}")

# 2) extra random groups of 4 for the demo (any 4 users with >= 10 past ratings, the same condition as group users)
N_RANDOM = 200
eligible = np.array(sorted(u for u, n in trva.groupby("user_id").size().items() if n >= 10))
rs = np.random.default_rng(SEED + 7)
for k in range(N_RANDOM):
    users = [int(u) for u in rs.choice(eligible, 4, replace=False)]
    out.append(build_group(1001 + k, users, "random"))
print(f"random groups: {N_RANDOM}")

meta = dict(model="v1 (results_v1_frozen)", lam=lam, percentile=final["stage2"]["percentile"],
            min_ratings_filter=min_count, groups=len(out),
            official=sum(g["kind"] == "official" for g in out), random=sum(g["kind"] == "random" for g in out))
js = "window.GROUP_DATA = " + json.dumps(dict(meta=meta, groups=out), ensure_ascii=False) + ";\n"
(ROOT / "frontend" / "data.js").write_text(js, encoding="utf-8")
print(f"saved frontend/data.js ({len(out)} groups)")
