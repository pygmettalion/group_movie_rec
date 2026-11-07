"""Demo: recommend one movie to 4 people, like a real recommendation.

Usage: RESULTS_DIR=results_v1_frozen python 05_demo.py [group_id]   (default 1)

How the candidate set is built
  all MovieLens movies
  -> remove movies any of the 4 has already rated (based on the full Train/Validation/Test history)
  -> keep only movies the final model can score (if the structure uses MF, only movies MF was trained on; unseen movies are not handled ad hoc)
  -> reliability filter: for movies with very few training ratings, the MF factors are close to 'guesses'.
     Splitting the Validation error by each movie's number of training ratings shows that movies with 20 or fewer ratings have large errors and inflated scores.
     Rule: keep only 'rating-count bins whose Validation RMSE is within 1.1x of the overall mean' -> the cutoff count is computed from the data by the code
  -> compute the Final Score for every candidate -> recommend the top one
* The original data has no movie titles, so movies are shown by movie_id and genres (titles are shown if movies.dat is added).
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, ROOT, SEED, load_json, load_movielens, movie_genre_table  # noqa: E402
from group import risk_proba  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

gid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
final = load_json("FINAL_FROZEN_CONFIG.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
r, groups = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
gt = movie_genre_table(r)
s1 = Stage1Predictor(final["stage1"], gt, seed=SEED).fit(trva)

row = groups[groups.group_id == gid].iloc[0]
users = [int(row[f"user_id_{i}"]) for i in range(1, 5)]
seen = set(r[r.user_id.isin(users)].movie_id)
all_movies = sorted(set(trva.movie_id))
# reliability-filter cutoff: look at the Validation error of the Train-only model (saved in step 02) by training-rating-count bins
tr = r[r.split == "train"]
v = pd.read_csv(RESULTS / "val_stage1_predictions.csv")
v["n"] = v.movie_id.map(tr.groupby("movie_id").size())
EDGES = [0, 5, 10, 20, 50, 100, 200, 500, 10**9]
v["bucket"] = pd.cut(v.n, EDGES)
by = v.groupby("bucket", observed=True).apply(lambda d: np.sqrt(((d.rating_norm - d.pred) ** 2).mean()))
overall = np.sqrt(((v.rating_norm - v.pred) ** 2).mean())
bad = [b for b, e in by.items() if e > 1.1 * overall]
min_count = int(max(b.right for b in bad)) + 1 if bad else 1
print(f"신뢰도 필터: Validation 기준 학습 평점 {min_count}개 이상인 영화만 후보 (구간별 RMSE: "
      + ", ".join(f"{b.left}-{b.right if b.right < 10**9 else '+'}: {e:.3f}" for b, e in by.items()) + ")")
n_trva = trva.groupby("movie_id").size()
cand = [m for m in all_movies if m not in seen and s1.mf_knows_item([m])[0] and gt[m].sum() > 0
        and n_trva.get(m, 0) >= min_count]
print(f"그룹 {gid} 사용자 {users}: 이미 본 영화 {len(seen)}편 제외 -> 후보 {len(cand)}편")

pred = np.column_stack([s1.predict([u] * len(cand), cand) for u in users])
th = user_thresholds(trva, final["stage2"]["percentile"])
thr = np.tile(th.thr.loc[users].to_numpy(), (len(cand), 1))
share = np.tile(th.share.loc[users].to_numpy(), (len(cand), 1))
R = risk_proba(risk, pred, thr, share, np.array(cand), movie_stats(trva))
E = pred.mean(1)
# final model = v1 formula. Passing 'v4' as the second argument compares with the v4 experiment formula (mean + 0.5*min - 3.0*risk).
v4 = ROOT / "results_v4" / "FINAL_FROZEN_CONFIG.json"
if len(sys.argv) > 2 and sys.argv[2] == "v4" and v4.exists():
    c4 = json.load(open(v4, encoding="utf-8"))
    F = E + c4["b"] * pred.min(1) - c4["a"] * R
    final["lam"] = f"{c4['a']} (+ 최저 예측 x {c4['b']})"
    tag = "_v4"   # v4 comparison output is saved to a separate file (so the v1 demo result is not overwritten)
else:
    F = E - final["lam"] * R
    tag = ""
genre_str = r.drop_duplicates("movie_id").set_index("movie_id").genres
titles = {}
if (ROOT / "data_extra" / "ml-1m" / "movies.dat").exists():   # show titles if the movies.dat obtained for v2 is available
    mv = pd.read_csv(ROOT / "data_extra" / "ml-1m" / "movies.dat", sep="::", engine="python",
                     names=["movie_id", "title", "g"], encoding="latin1")
    titles = dict(zip(mv.movie_id, mv.title))
out = pd.DataFrame({"movie_id": cand, "title": [titles.get(m, "") for m in cand],
                    "genres": [genre_str[m] for m in cand], "expected_satisfaction": E,
                    "misery_risk": R, "final_score": F,
                    **{f"pred_user{u}_stars": 1 + 4 * np.clip(pred[:, j], 0, 1) for j, u in enumerate(users)}})
out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
out.to_csv(RESULTS / f"demo_group{gid}{tag}.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nlambda = {final['lam']}  (Final = 예상만족 - lambda x Strong Dislike 위험)")
print(out.head(5).round(3).to_string())
avg_pick = out.loc[out.expected_satisfaction.idxmax()]
print(f"\n최종 추천: movie {int(out.movie_id[0])} {out.title[0]} ({out.genres[0]})  예상만족 {out.expected_satisfaction[0]:.3f}, "
      f"Strong Dislike 위험 {out.misery_risk[0]:.3f}")
print(f"참고 - 평균만 봤다면: movie {int(avg_pick.movie_id)} {avg_pick.title} ({avg_pick.genres})  예상만족 "
      f"{avg_pick.expected_satisfaction:.3f}, Strong Dislike 위험 {avg_pick.misery_risk:.3f}")
