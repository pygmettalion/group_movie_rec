"""[추가 실험] MovieLens 10M 에서 1차 모델을 '숫자 그대로' 옮겨 한 번 채점.

- Stage 1: 1차 설정(ALS k=64, reg=0.02, 8회 / Content k_mean=20, lam_user=100 / 평점 수별 섞기 비율) 그대로,
  10M 의 Train+Validation 으로 학습만 다시.
- 극혐선: 각자 Train+Validation 평점의 하위 5% (같은 규칙)
- 위험 모델: 1M 에서 학습한 1차 위험 모델 파일을 '그대로' 사용 (다시 학습하지 않음)
- 점수 식: 1차 (평균 - 0.75 x 위험), 4차 (평균 + 0.5 x 최저 - 3.0 x 위험)
- 비교: 기존 추천·그룹 추천 규칙 전부. 시험 그룹은 51 에서 별점 안 보고 봉인한 것.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import movie_genre_table  # noqa: E402
from group import paired_bootstrap  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import indiv_features, movie_stats, user_thresholds  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_ml10m"
cfg1 = json.load(open(ROOT / "results_v1_frozen" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
cfg4 = json.load(open(ROOT / "results_v4" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
with open(ROOT / "results_v1_frozen" / "risk_model.pkl", "rb") as f:
    R1 = pickle.load(f)
r = pd.read_pickle(OUT / "ratings.pkl")
trva, te = r[r.split != "test"], r[r.split == "test"]
t0 = time.time()
s1 = Stage1Predictor(cfg1["stage1"], movie_genre_table(r), seed=42).fit(trva)
print(f"학습 완료 {time.time() - t0:.0f}초")
th = user_thresholds(trva, 5)
mstats = movie_stats(trva)
cnt = trva.groupby("movie_id").size()
agg = trva.groupby("movie_id").rating_norm.agg(["sum", "count"])
top_rated = (agg["sum"] + 20 * trva.rating_norm.mean()) / (agg["count"] + 20)
truth = te.set_index(["user_id", "movie_id"]).rating_norm
truth = truth[~truth.index.duplicated()]

# (참고) 10M Test 기간 개인 예측 오차
tw = te[te.user_id.isin(trva.user_id) & te.movie_id.isin(trva.movie_id)].sample(200000, random_state=0)
rm = float(np.sqrt(np.mean((tw.rating_norm - s1.predict(tw.user_id.values, tw.movie_id.values)) ** 2)) * 4.5)
um = float(np.sqrt(np.mean((tw.rating_norm - tw.user_id.map(trva.groupby("user_id").rating_norm.mean())) ** 2)) * 4.5)
print(f"개인 예측 RMSE(별점): 모델 {rm:.3f} / 사용자 평균만 {um:.3f}")


def awm(pred, thr):
    ok = (pred > thr[None, :]).all(1)
    m = pred.mean(1)
    return int(np.argmax(np.where(ok, m, -np.inf))) if ok.any() else int(np.argmax(m))


def evaluate(groups, tag):
    rows = []
    for g in groups.itertuples(index=False):
        users = [g.user_id_1, g.user_id_2, g.user_id_3, g.user_id_4]
        movies = [int(x) for x in g.movie_ids.split("|")]
        pred = np.column_stack([s1.predict([u] * len(movies), movies) for u in users])
        true = np.array([[truth[(u, m)] for u in users] for m in movies])
        thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
        pi = np.column_stack([R1["model"].predict_proba(indiv_features(pred[:, j], np.full(len(movies), thr[j]),
                                                                       np.full(len(movies), share[j]), np.array(movies),
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
                 "인기순": [int(np.argmax([cnt.get(m, 0) for m in movies]))],
                 "평점순": [int(np.argmax([top_rated.get(m, 0) for m in movies]))],
                 "아무거나": list(range(len(movies))), "정답 아는 최선": [int(np.argmax(true.mean(1)))]}
        for name, idx in picks.items():
            o = [(true[i].mean(), true[i].min(), float((true[i] <= thr).any())) for i in idx]
            rows.append(dict(set=tag, gid=g.group_id, n=len(movies), method=name, avg=np.mean([x[0] for x in o]),
                             min=np.mean([x[1] for x in o]), sd=np.mean([x[2] for x in o])))
    return pd.DataFrame(rows)


df = pd.concat([evaluate(pd.read_csv(OUT / "TEST_groups_sealed.csv"), "5편 이상"),
                evaluate(pd.read_csv(OUT / "TEST_groups10_sealed.csv"), "10편 이상")])
df.to_csv(OUT / "ML10M_per_group.csv", index=False)
res = {"individual_rmse_stars": dict(model=rm, user_mean=um)}
for tag, d in df.groupby("set", sort=False):
    s = d.groupby("method", sort=False)[["avg", "min", "sd"]].mean()
    print(f"\n[10M Test | 공통 {tag} | {d.gid.nunique()}그룹, 후보 평균 {d.groupby('gid').n.first().mean():.1f}편]\n", s.round(4).to_string())
    g = {m: d[d.method == m].sort_values("gid").reset_index(drop=True) for m in s.index}
    boot = {f"우리 1차 - {b}": paired_bootstrap(g["우리 1차"], g[b]) for b in ["Average", "Least Misery", "AWM", "한 사람 계정", "인기순"]}
    boot["우리 4차 식 - Average"] = paired_bootstrap(g["우리 4차 식"], g["Average"])
    boot["우리 4차 식 - 우리 1차"] = paired_bootstrap(g["우리 4차 식"], g["우리 1차"])
    for k, v in boot.items():
        print(f"  {k}: " + ", ".join(f"{m} {x['diff']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]" for m, x in v.items()))
    res[tag] = dict(summary=s.reset_index().to_dict("records"), bootstrap=boot)
json.dump(res, open(OUT / "ML10M_RESULTS.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
