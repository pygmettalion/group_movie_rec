"""[additional experiment] Prepare MovieLens 10M: build data and test groups with 'exactly the same rules' as 1M.

Why?
  In 1M, groups where '4 people share many watched movies' are rare, so it was hard to show performance when there are many candidates.
  The same method is applied from scratch to 10M (about 70k users). No number chosen on 1M is changed.

Rules (same as the team preprocessing code prepare_team_data.py)
  - time-ordered split: by rating time, first 70% = Train, next 15% = Validation, last 15% = Test
  - group users: at least 10 Train+Validation ratings
  - rating normalization: 0.5~5 stars -> (rating-0.5)/4.5 (0~1)
  - Validation groups: >= 5 common movies in the Validation period (pseudo-groups for training the risk model are separate)
  - Test groups: >= 5 common movies in the Test period, with the >= 10 subset marked separately (sealed without looking at ratings)

Input: data_extra/ml-10m/ (ratings.dat, movies.dat)   Output: results_ml10m/
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_ml10m"
OUT.mkdir(exist_ok=True)
SEED, MIN_HISTORY, MIN_COMMON, N_GROUPS = 42, 10, 5, 1500


def load(src):
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        rname = next(n for n in z.namelist() if n.endswith("ratings.dat"))
        mname = next(n for n in z.namelist() if n.endswith("movies.dat"))
        rf, mf = z.open(rname), z.open(mname)
    else:
        rf, mf = open(src / "ratings.dat", "rb"), open(src / "movies.dat", "rb")
    r = pd.read_csv(rf, sep="::", engine="python", names=["user_id", "movie_id", "rating_raw", "timestamp"])
    m = pd.read_csv(mf, sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="latin1")
    return r, m


src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_extra" / "ml-10m"
r, movies = load(src)
r["rating_norm"] = (r.rating_raw - 0.5) / 4.5
movies["genres"] = movies.genres.str.replace("Children", "Children's", regex=False)   # match the 1M notation
r = r.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
ts = np.sort(r.timestamp.to_numpy())
v_start, t_start = ts[len(ts) * 70 // 100], ts[len(ts) * 85 // 100]
r["split"] = np.where(r.timestamp < v_start, "train", np.where(r.timestamp < t_start, "validation", "test"))
r.to_pickle(OUT / "ratings.pkl")
movies.to_pickle(OUT / "movies.pkl")
print(f"평점 {len(r):,} / 사용자 {r.user_id.nunique():,} / 영화 {r.movie_id.nunique():,}")
print(r.split.value_counts().to_dict(), "| Validation 시작", pd.to_datetime(v_start, unit="s"), "Test 시작", pd.to_datetime(t_start, unit="s"))


MAX_PER_USER = 10   # cap on the number of groups one person can join (so a few heavy viewers do not dominate the test)


def find_groups(period, exclude_users, n_max, rng, min_common):
    """Groups of 4 people who share at least min_common movies in the given period (ratings are not used).
    People are visited in random order; for each, the people with the most co-watched movies are grouped into 4, and the number of groups per person is capped at MAX_PER_USER."""
    from collections import Counter
    hist = r[r.timestamp < (v_start if period == "validation" else t_start)].groupby("user_id").size()
    ok = set(hist[hist >= MIN_HISTORY].index) - exclude_users
    known = set(r[r.split == "train"].movie_id) if period == "validation" else set(r[r.split != "test"].movie_id)
    p = r[(r.split == period) & r.user_id.isin(ok) & r.movie_id.isin(known)][["user_id", "movie_id"]]
    sets = {u: set(d.movie_id) for u, d in p.groupby("user_id") if len(d) >= min_common}
    by_movie = {}
    for u, ms in sets.items():
        for m in ms:
            by_movie.setdefault(m, []).append(u)
    used, found = Counter(), set()
    seeds = list(sets)
    rng.shuffle(seeds)
    for a in seeds:
        if len(found) >= n_max:
            break
        if used[a] >= MAX_PER_USER:
            continue
        co = Counter()
        for m in sets[a]:
            co.update(by_movie[m])
        partners = [b for b, c in co.items() if b != a and c >= min_common and used[b] < MAX_PER_USER]
        if len(partners) < 3:
            continue
        for _ in range(30):
            trio = tuple(rng.choice(partners, 3, replace=False))
            g = tuple(sorted((a, *trio)))
            if g in found or any(used[x] >= MAX_PER_USER for x in g):
                continue
            if len(sets[a] & sets[trio[0]] & sets[trio[1]] & sets[trio[2]]) >= min_common:
                found.add(g)
                used.update(g)
                break
    found = sorted(found)
    rows = []
    for gid, g in enumerate(found, 1):
        common = sorted(set.intersection(*[sets[u] for u in g]))
        rows.append(dict(group_id=gid, user_id_1=g[0], user_id_2=g[1], user_id_3=g[2], user_id_4=g[3],
                         n_common=len(common), movie_ids="|".join(map(str, common))))
    return pd.DataFrame(rows), len(sets)


rng = np.random.default_rng(SEED)
test_groups, n_found = find_groups("test", set(), N_GROUPS, rng, MIN_COMMON)
test_groups.to_csv(OUT / "TEST_groups_sealed.csv", index=False)
print(f"Test 그룹 봉인: 후보 사용자 {n_found:,}명, 그룹 {len(test_groups)}개, 후보 평균 {test_groups.n_common.mean():.1f}편, "
      f"10편 이상 {int((test_groups.n_common >= 10).sum())}개, 사용자 {len(set(test_groups.iloc[:, 1:5].values.ravel()))}명")

# groups with many candidates (>= 10 common) are also sealed separately — to check performance 'when there are many movies to choose from'
groups10, n10 = find_groups("test", set(), N_GROUPS, np.random.default_rng(SEED + 1), 10)
groups10.to_csv(OUT / "TEST_groups10_sealed.csv", index=False)
print(f"Test 그룹(10편 이상) 봉인: 후보 사용자 {n10:,}명, 그룹 {len(groups10)}개, 후보 평균 {groups10.n_common.mean():.1f}편, "
      f"사용자 {len(set(groups10.iloc[:, 1:5].values.ravel()))}명")
