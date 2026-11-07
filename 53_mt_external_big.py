"""[enlarging the external test] Build more external test groups from the raw MovieTweetings data and score the v1 model once.

Follows the team preprocessing rules (prepare_team_data.py choose_groups/build_bundle):
  - only 'movies that also exist in MovieLens' are test movies (IMDb->MovieLens link via links.csv + present in MovieLens Train+Validation)
  - movies rated from a user's 10th linked rating onward = movies that can become test candidates
  - 4 people share >= 5 test movies
  - each person's history = only ratings 'before' the earliest rating time among that group's test movies (needs >= 10 linked past ratings)
  - ratings 0~10 -> /10
One change: the team code puts each person in only one group, giving just 44 groups -> at most 10 groups per person.
The 176 people of the existing 44 groups are excluded. The v1 model (results_v1_frozen) is used with the same numbers and scored once.
"""
import json
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import ROOT, SEED, genre_vector, load_movielens, load_movietweetings, movie_genre_table  # noqa: E402
from group import paired_bootstrap  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import indiv_features, movie_stats, user_thresholds  # noqa: E402

OUT = ROOT / "results_mt_big"
OUT.mkdir(exist_ok=True)
RAW = ROOT.parent / "ext" / "mt" / "MovieTweetings-master" / "latest"
MIN_HIST, MIN_COMMON, MAX_PER_USER, N_MAX = 10, 5, 10, 1000

r, _ = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
known_ml = set(trva.movie_id)
links = pd.read_csv(ROOT / "data_extra" / "ml-latest-small" / "links.csv")
mapping = {int(i): int(m) for m, i in zip(links.movieId, links.imdbId) if int(m) in known_ml}
cw = pd.read_csv(ROOT / "data" / "02_MovieTweetings" / "movie_id_crosswalk.csv", encoding="utf-8-sig")
mapping.update({int(t[2:]): int(m) for t, m in zip(cw.movie_id, cw.movielens_movie_id) if int(m) in known_ml})

rt = pd.read_csv(RAW / "ratings.dat", sep="::", engine="python", names=["user_id", "imdb", "score", "ts"])
mv = pd.read_csv(RAW / "movies.dat", sep="::", engine="python", names=["imdb", "title", "genres"], encoding="utf-8")
mt_genre = dict(zip(mv.imdb, mv.genres))
old_mt, old_groups = load_movietweetings()
old_users = set(old_groups[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())
rt = rt[~rt.user_id.isin(old_users)]
rt["ml"] = rt.imdb.map(mapping)
print(f"원본 평점 {len(rt):,}, 사용자 {rt.user_id.nunique():,}, MovieLens 연결 평점 {rt.ml.notna().sum():,}")

# test-candidate movies per user (from the 10th linked rating onward)
cand = {}
for u, d in rt[rt.ml.notna()].sort_values("ts").groupby("user_id"):
    if len(d) < MIN_HIST + MIN_COMMON:
        continue
    cutoff = d.ts.iloc[MIN_HIST - 1]
    s = set(d[d.ts > cutoff].ml.astype(int))
    if len(s) >= MIN_COMMON:
        cand[u] = s
by_movie = {}
for u, s in cand.items():
    for m in s:
        by_movie.setdefault(m, []).append(u)
rng = np.random.default_rng(SEED)
seeds = list(cand)
rng.shuffle(seeds)
used, found = Counter(), []
seen = set()
user_rows = {u: d.sort_values("ts") for u, d in rt[rt.user_id.isin(cand)].groupby("user_id")}
for a in seeds:
    if len(found) >= N_MAX or used[a] >= MAX_PER_USER:
        continue
    co = Counter()
    for m in cand[a]:
        co.update(by_movie[m])
    partners = [b for b, c in co.items() if b != a and c >= MIN_COMMON and used[b] < MAX_PER_USER]
    if len(partners) < 3:
        continue
    for _ in range(30):
        trio = tuple(int(x) for x in rng.choice(partners, 3, replace=False))
        g = tuple(sorted((a, *trio)))
        if g in seen or any(used[x] >= MAX_PER_USER for x in g):
            continue
        common = sorted(cand[a] & cand[trio[0]] & cand[trio[1]] & cand[trio[2]])
        if len(common) < MIN_COMMON:
            continue
        # each person's history = ratings 'before' the earliest rating of that group's test movies (check >= 10 linked past ratings: only times are used)
        ok = True
        for u in g:
            d = user_rows[u]
            cut = d[d.ml.isin(common)].ts.min()
            if (d[d.ts < cut].ml.notna()).sum() < MIN_HIST:
                ok = False
                break
        seen.add(g)
        if ok:
            found.append((g, common))
            used.update(g)
            break
print(f"새 외부 시험 그룹 {len(found)}개, 사용자 {len(set(x for g, _ in found for x in g))}명, "
      f"후보 평균 {np.mean([len(c) for _, c in found]):.1f}편")
pd.DataFrame([dict(group_id=i, users="|".join(map(str, g)), movies="|".join(map(str, c)))
              for i, (g, c) in enumerate(found, 1)]).to_csv(OUT / "MT_BIG_groups_sealed.csv", index=False)

# ---------------------------------------------------------------- score once with the v1 model
cfg1 = json.load(open(ROOT / "results_v1_frozen" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
cfg4 = json.load(open(ROOT / "results_v4" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
with open(ROOT / "results_v1_frozen" / "risk_model.pkl", "rb") as f:
    R1 = pickle.load(f)
gt_ml = movie_genre_table(r)
s1 = Stage1Predictor(cfg1["stage1"], gt_ml, seed=SEED).fit(trva)
mstats = movie_stats(trva)
cnt = trva.groupby("movie_id").size()
GENRE_OK = set(["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])


def key_and_genre(imdb, ml):
    """Linked movies use the MovieLens id and genres; otherwise the 'tt' id and MovieTweetings genres (only those within the 18) — same as the team rule."""
    if not pd.isna(ml):
        return int(ml), gt_ml.get(int(ml))
    raw = str(mt_genre.get(imdb, "")).split("|")
    return f"tt{imdb:07d}", genre_vector("|".join(x for x in raw if x in GENRE_OK))


def awm(pred, thr):
    ok = (pred > thr[None, :]).all(1)
    m = pred.mean(1)
    return int(np.argmax(np.where(ok, m, -np.inf))) if ok.any() else int(np.argmax(m))


rows = []
for gid, (g, common) in enumerate(found, 1):
    cols, thr, share, true = [], [], [], []
    for u in g:
        d = user_rows[u]
        cut = d[d.ml.isin(common)].ts.min()
        h = d[d.ts < cut]
        keys, gtab = [], {}
        for imdb, ml in zip(h.imdb, h.ml):
            k, v = key_and_genre(imdb, ml)
            keys.append(k)
            gtab[k] = v
        for m in common:
            gtab[m] = gt_ml.get(m)
        hy = (h.score / 10.0).to_numpy()
        cols.append(s1.predict_new_user(keys, hy, common, genre_table=gtab)[0])
        tu = user_thresholds(pd.DataFrame({"user_id": u, "rating_norm": hy}), 5)
        thr.append(float(tu.thr.iloc[0]))
        share.append(float(tu.share.iloc[0]))
        last = d[d.ml.isin(common)].drop_duplicates("ml", keep="last").set_index("ml").score / 10.0
        true.append([float(last[m]) for m in common])
    pred, true, thr, share = np.column_stack(cols), np.array(true).T, np.array(thr), np.array(share)
    pi = np.column_stack([R1["model"].predict_proba(indiv_features(pred[:, j], np.full(len(common), thr[j]),
                                                                   np.full(len(common), share[j]), np.array(common),
                                                                   mstats)[R1["features"]])[:, 1] for j in range(4)])
    rk = 1 - np.prod(1 - pi, axis=1)
    mean = pred.mean(1)
    picks = {"우리 1차": [int(np.argmax(mean - cfg1["lam"] * rk))],
             "우리 4차 식": [int(np.argmax(mean + cfg4["b"] * pred.min(1) - cfg4["a"] * rk))],
             "Average": [int(np.argmax(mean))], "Least Misery": [int(np.argmax(pred.min(1) + 1e-9 * mean))],
             "AWM": [awm(pred, thr)], "Borda": [int(np.argmax(np.argsort(np.argsort(pred, 0), 0).sum(1) + 1e-9 * mean))],
             "Multiplicative": [int(np.argmax(np.prod(np.clip(pred, 1e-3, None), 1)))],
             "Most Pleasure": [int(np.argmax(pred.max(1) + 1e-9 * mean))],
             "한 사람 계정": [int(np.argmax(pred[:, j])) for j in range(4)],
             "인기순": [int(np.argmax([cnt.get(m, 0) for m in common]))],
             "아무거나": list(range(len(common))), "정답 아는 최선": [int(np.argmax(true.mean(1)))]}
    for name, idx in picks.items():
        o = [(true[i].mean(), true[i].min(), float((true[i] <= thr).any())) for i in idx]
        rows.append(dict(gid=gid, n=len(common), method=name, avg=np.mean([x[0] for x in o]),
                         min=np.mean([x[1] for x in o]), sd=np.mean([x[2] for x in o])))
df = pd.DataFrame(rows)
df.to_csv(OUT / "MT_BIG_per_group.csv", index=False)
s = df.groupby("method", sort=False)[["avg", "min", "sd"]].mean()
print(f"\n[MovieTweetings 확대 외부 시험 {df.gid.nunique()}그룹]\n", s.round(4).to_string())
gg = {m: df[df.method == m].sort_values("gid").reset_index(drop=True) for m in s.index}
boot = {f"우리 1차 - {b}": paired_bootstrap(gg["우리 1차"], gg[b])
        for b in ["Average", "Least Misery", "AWM", "Borda", "한 사람 계정", "인기순"]}
boot["우리 4차 식 - 우리 1차"] = paired_bootstrap(gg["우리 4차 식"], gg["우리 1차"])
for k, v in boot.items():
    print(f"  {k}: " + ", ".join(f"{m} {x['diff']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]" for m, x in v.items()))
json.dump(dict(n_groups=int(df.gid.nunique()), summary=s.reset_index().to_dict("records"), bootstrap=boot),
          open(OUT / "MT_BIG_RESULTS.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
