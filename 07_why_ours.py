"""'Why use our recommender?' — compare common recommendation methods with the v1 final model.

Compared methods (all fixed rules; nothing is tuned on test data)
  - single-account recommendation (Host): like Netflix, the top pick of the one person 'who pressed play'. Averaged over each of the 4 being the host.
  - Popular: the movie watched most often in the past
  - Top-rated: the movie with the highest past average rating (movies with few ratings are shrunk toward the global mean)
  - Average (maximize the mean): the movie with the highest mean of the 4 predictions = a typical group recommender that only targets the 'best case'
  - our model (Risk-Aware, v1 frozen): mean - 0.75 x Strong Dislike risk
Scoring: fresh holdouts 1 and 2, 1,000 groups (v1 was frozen before either was seen).
Key metrics: 'share of groups where nobody strongly disliked the pick' (= 1 - Strong Dislike rate), minimum satisfaction. Average satisfaction is checked as well.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import ROOT, SEED, group_rows, load_movielens, movie_genre_table  # noqa: E402
from group import build_case, outcome, paired_bootstrap  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

D = ROOT / "results_v1_frozen"
cfg = json.load(open(D / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
with open(D / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
r, _ = load_movielens()
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
s1 = Stage1Predictor(cfg["stage1"], movie_genre_table(r), seed=SEED).fit(trva)
th = user_thresholds(trva, cfg["stage2"]["percentile"])
mstats = movie_stats(trva)
truth = te.set_index(["user_id", "movie_id"]).rating_norm
cnt = trva.groupby("movie_id").size()
mean = trva.groupby("movie_id").rating_norm.agg(["sum", "count"])
top_rated = (mean["sum"] + 20 * trva.rating_norm.mean()) / (mean["count"] + 20)

def awm(pred, thr):
    """Average Without Misery: drop movies where someone's 'predicted score' is at or below their own Strong Dislike threshold, then take the top mean. If all are dropped, take the top mean."""
    ok = (pred > thr[None, :]).all(1)
    m = pred.mean(1)
    return int(np.argmax(np.where(ok, m, -np.inf))) if ok.any() else int(np.argmax(m))


rows, awm_trig = [], []
for hname, path in [("1번", ROOT / "results_v2" / "FRESH_HOLDOUT_groups.csv"), ("2번", ROOT / "results_v3" / "FRESH_HOLDOUT2_groups.csv")]:
    for gid, users, movies in group_rows(pd.read_csv(path), "test"):
        pred = np.column_stack([s1.predict([u] * len(movies), movies) for u in users])
        true = np.array([[truth[(u, m)] for u in users] for m in movies])
        thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
        c = build_case(gid, users, movies, pred, true, thr, share, risk, mstats)
        awm_trig.append(int(not (pred > thr[None, :]).all(1).all()))
        picks = {
            "Host (한 사람 계정)": [int(np.argmax(pred[:, j])) for j in range(4)],
            "Popular (인기순)": [int(np.argmax([cnt.get(m, 0) for m in movies]))],
            "Top-rated (평점순)": [int(np.argmax([top_rated.get(m, 0) for m in movies]))],
            "Average (평균 최대)": [int(np.argmax(c["exp_sat"]))],
            # --- classic rules from the group recommendation literature (Masthoff's group aggregation strategies) ---
            "Least Misery (최저점 최대)": [int(np.argmax(pred.min(1) + 1e-9 * pred.mean(1)))],
            "Most Pleasure (최고점 최대)": [int(np.argmax(pred.max(1) + 1e-9 * pred.mean(1)))],
            "Multiplicative (곱 최대)": [int(np.argmax(np.prod(np.clip(pred, 1e-3, None), axis=1)))],
            "Borda (순위 합)": [int(np.argmax(np.argsort(np.argsort(pred, axis=0), axis=0).sum(1) + 1e-9 * pred.mean(1)))],
            "AWM (불만 제외 후 평균)": [awm(pred, thr)],
            "Ours (Risk-Aware)": [int(np.argmax(c["exp_sat"] - cfg["lam"] * c["risk"]))],
            "Random (아무거나)": list(range(len(movies))),
        }
        for name, idx in picks.items():
            o = [outcome(c, i) for i in idx]
            rows.append(dict(holdout=hname, group=f"{hname}-{gid}", method=name,
                             **{k: float(np.mean([x[k] for x in o])) for k in ("avg", "min", "sd")}))
print(f"AWM 이 실제로 영화를 하나라도 뺀 그룹 비율: {np.mean(awm_trig):.3f}")
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results_v3" / "why_ours_per_group.csv", index=False)
order = ["Random (아무거나)", "Popular (인기순)", "Top-rated (평점순)", "Host (한 사람 계정)",
         "Most Pleasure (최고점 최대)", "Borda (순위 합)", "Multiplicative (곱 최대)", "Least Misery (최저점 최대)",
         "AWM (불만 제외 후 평균)", "Average (평균 최대)", "Ours (Risk-Aware)"]
summ = df.groupby("method")[["avg", "min", "sd"]].mean().reindex(order)
summ["nobody_hates"] = 1 - summ.sd
summ.to_csv(ROOT / "results_v3" / "why_ours_summary.csv")
print(summ.round(4).to_string())
g = {m: df[df.method == m].sort_values("group").reset_index(drop=True) for m in order}
out = {}
for b in order[:-1]:
    out[b] = paired_bootstrap(g["Ours (Risk-Aware)"], g[b])
    print(f"Ours - {b}: " + ", ".join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in out[b].items()))
# check whether Least Misery loses average satisfaction (evidence that explains the difference from our model)
lm_avg = paired_bootstrap(g["Least Misery (최저점 최대)"], g["Average (평균 최대)"])
out["(참고) Least Misery - Average"] = lm_avg
print("(참고) Least Misery - Average: " + ", ".join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in lm_avg.items()))
json.dump(dict(summary=summ.reset_index().to_dict("records"), bootstrap=out),
          open(ROOT / "results_v3" / "why_ours.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
