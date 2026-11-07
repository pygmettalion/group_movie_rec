"""[4차] 외부 데이터 시험: MovieTweetings 44그룹 (다른 서비스 사용자). 4차는 이미 고정됨 -> 한 번만 채점.

1차 외부 시험(04_final_test.py)과 같은 방식: MovieLens(Train+Validation)로 학습한 모델에
MovieTweetings 사용자의 과거 평점을 넣어(fold-in) 예측, 극혐선은 각자의 MovieTweetings 과거 평점으로 계산.
실행: python 43_v4_external.py
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import ROOT, SEED, group_rows, load_movielens, load_movietweetings, movie_genre_table  # noqa: E402
from group import build_case, paired_bootstrap  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

cfg1 = json.load(open(ROOT / "results_v1_frozen" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
cfg4 = json.load(open(ROOT / "results_v4" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
with open(ROOT / "results_v1_frozen" / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
assert cfg4["combo"] == "S1+RV1"   # 4차 = 1차 예측 + 1차 위험 모델 + 새 점수 식
r, _ = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
s1 = Stage1Predictor(cfg1["stage1"], movie_genre_table(r), seed=SEED).fit(trva)
mstats = movie_stats(trva)

mt, mt_groups = load_movietweetings()
hist, test = mt[mt.split == "history"], mt[mt.split == "test"]
gt_mt = {int(k) if k.isdigit() else k: v for k, v in movie_genre_table(mt).items()}
th = user_thresholds(hist, 5)
truth = test.assign(mid=test.movie_id.astype(int)).set_index(["user_id", "mid"]).rating_norm
hb = {u: d for u, d in hist.groupby("user_id")}
cnt = trva.groupby("movie_id").size()

rows = []
for gid, users, movies in group_rows(mt_groups, "test"):
    pred = np.column_stack([s1.predict_new_user([int(m) if m.isdigit() else m for m in hb[u].movie_id],
                                                hb[u].rating_norm.values, movies, genre_table=gt_mt)[0] for u in users])
    true = np.array([[truth[(u, m)] for u in users] for m in movies])
    thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
    c = build_case(gid, users, movies, pred, true, thr, share, risk, mstats)
    mean, rk = pred.mean(1), c["risk"]
    picks = {"4차 (최종)": [int(np.argmax(mean + cfg4["b"] * pred.min(1) - cfg4["a"] * rk))],
             "1차 Risk-Aware": [int(np.argmax(mean - cfg1["lam"] * rk))],
             "평균 최대": [int(np.argmax(mean))], "Least Misery": [int(np.argmax(pred.min(1) + 1e-9 * mean))],
             "한 사람 계정": [int(np.argmax(pred[:, j])) for j in range(4)],
             "인기순": [int(np.argmax([cnt.get(m, 0) for m in movies]))],
             "아무거나": list(range(len(movies))), "정답 아는 최선": [int(np.argmax(true.mean(1)))]}
    for name, idx in picks.items():
        o = [(true[i].mean(), true[i].min(), float((true[i] <= thr).any())) for i in idx]
        rows.append(dict(gid=gid, method=name, avg=np.mean([x[0] for x in o]), min=np.mean([x[1] for x in o]),
                         sd=np.mean([x[2] for x in o])))
df = pd.DataFrame(rows)
summ = df.groupby("method", sort=False)[["avg", "min", "sd"]].mean()
print(f"[MovieTweetings 외부 시험 {df.gid.nunique()}그룹, 후보 평균 5.2편]\n", summ.round(4).to_string())
g = {m: df[df.method == m].sort_values("gid").reset_index(drop=True) for m in summ.index}
boot = {f"4차 - {b}": paired_bootstrap(g["4차 (최종)"], g[b]) for b in ["1차 Risk-Aware", "평균 최대", "한 사람 계정"]}
for k, v in boot.items():
    print(f"  {k}: " + ", ".join(f"{m} {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for m, d in v.items()))
print("4차와 1차가 다른 영화를 고른 그룹 수:",
      int((g["4차 (최종)"].avg != g["1차 Risk-Aware"].avg).sum() + 0))
df.to_csv(ROOT / "results_v4" / "EXTERNAL_MT_per_group.csv", index=False)
json.dump(dict(summary=summ.reset_index().to_dict("records"), bootstrap=boot),
          open(ROOT / "results_v4" / "V4_EXTERNAL_MT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
