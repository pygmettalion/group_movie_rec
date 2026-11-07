"""[version 4] Final scoring: v1 vs v4 once on holdout 3, which nobody has seen (1,000 groups, no person shared with holdouts 1 and 2).

Pre-registered rule: adopt v4 if it reduces Strong Dislike more than v1 without clearly losing average satisfaction (95% interval below 0).
Run: python 42_v4_final.py
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import ROOT, SEED, group_rows, load_movielens, movie_genre_table  # noqa: E402
from group import paired_bootstrap  # noqa: E402
from risk import indiv_features, movie_stats, user_thresholds  # noqa: E402
from risk_v2 import RiskContext, indiv_features_v2  # noqa: E402
from stage1_v3 import make_stage1  # noqa: E402

OUT = ROOT / "results_v4"
cfg4 = json.load(open(OUT / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
cfg1 = json.load(open(ROOT / "results_v1_frozen" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
with open(OUT / "risk_model.pkl", "rb") as f:
    R4 = pickle.load(f)
with open(ROOT / "results_v1_frozen" / "risk_model.pkl", "rb") as f:
    R1 = pickle.load(f)
print("4차:", cfg4["combo"], cfg4["rule"], cfg4["a"], cfg4["b"])

r, _ = load_movielens()
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
mstats = movie_stats(trva)
th = user_thresholds(trva, 5)
ctx = RiskContext(trva, th.thr, genre_lookup, 3.0, trva, 50.0)
s1 = make_stage1(cfg1["stage1"], gt, SEED).fit(trva)
s4 = s1 if cfg4["stage1_from"] == "results_v1_frozen" else make_stage1(
    json.load(open(ROOT / cfg4["stage1_from"] / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))["stage1"], gt, SEED,
    results_dir=ROOT / cfg4["stage1_from"]).fit(trva)
cnt = trva.groupby("movie_id").size()
agg = trva.groupby("movie_id").rating_norm.agg(["sum", "count"])
top_rated = (agg["sum"] + 20 * trva.rating_norm.mean()) / (agg["count"] + 20)

groups = group_rows(pd.read_csv(OUT / "FRESH_HOLDOUT3_groups.csv"), "test")
rows = pd.DataFrame([(gid, j, u, m) for gid, users, movies in groups for m in movies for j, u in enumerate(users)],
                    columns=["gid", "j", "user_id", "movie_id"])
rows["y"] = te.set_index(["user_id", "movie_id"]).rating_norm.reindex(list(zip(rows.user_id, rows.movie_id))).to_numpy()
rows["thr"], rows["share"] = rows.user_id.map(th.thr), rows.user_id.map(th.share)
rows["p1"] = s1.predict(rows.user_id.values, rows.movie_id.values)
rows["p4"] = s4.predict(rows.user_id.values, rows.movie_id.values) if s4 is not s1 else rows.p1
X1 = indiv_features(rows.p1.values, rows.thr.values, rows.share.values, rows.movie_id.values, mstats)
rows["r1"] = R1["model"].predict_proba(X1[R1["features"]])[:, 1]
if R4.get("kind") == "indiv_v2":
    X4 = indiv_features_v2(rows.p4.values, rows.thr.values, rows.share.values, rows.movie_id.values,
                           rows.user_id.values, mstats, ctx)
else:
    X4 = indiv_features(rows.p4.values, rows.thr.values, rows.share.values, rows.movie_id.values, mstats)
rows["r4"] = R4["model"].predict_proba(X4[R4["features"]])[:, 1]

out = []
for gid, d in rows.groupby("gid", sort=False):
    movies = d.movie_id.unique().tolist()
    M = lambda c: d.pivot(index="movie_id", columns="j", values=c).loc[movies].to_numpy()  # noqa: E731
    y, p1, p4 = M("y"), M("p1"), M("p4")
    thr = d.drop_duplicates("j").sort_values("j").thr.to_numpy()
    k1 = 1 - np.prod(1 - M("r1"), axis=1)
    k4 = 1 - np.prod(1 - M("r4"), axis=1)
    m4 = p4.mean(1)
    a, b = cfg4["a"], cfg4["b"]
    if cfg4["rule"] == "LIN":
        i4 = int(np.argmax(m4 - a * k4))
    elif cfg4["rule"] == "LINMIN":
        i4 = int(np.argmax(m4 + b * p4.min(1) - a * k4))
    else:
        ok = k4 <= a
        i4 = int(np.argmax(np.where(ok, m4, -np.inf))) if ok.any() else int(np.argmin(k4))
    picks = {"4차 (최종 후보)": [i4], "1차 Risk-Aware": [int(np.argmax(p1.mean(1) - cfg1["lam"] * k1))],
             "평균 최대": [int(np.argmax(p1.mean(1)))], "Least Misery": [int(np.argmax(p1.min(1) + 1e-9 * p1.mean(1)))],
             "한 사람 계정": [int(np.argmax(p1[:, j])) for j in range(4)],
             "인기순": [int(np.argmax([cnt.get(m, 0) for m in movies]))],
             "평점순": [int(np.argmax([top_rated.get(m, 0) for m in movies]))],
             "아무거나": list(range(len(movies))), "정답 아는 최선": [int(np.argmax(y.mean(1)))]}
    for name, idx in picks.items():
        o = [(y[i].mean(), y[i].min(), float((y[i] <= thr).any())) for i in idx]
        out.append(dict(gid=gid, method=name, avg=np.mean([x[0] for x in o]), min=np.mean([x[1] for x in o]),
                        sd=np.mean([x[2] for x in o])))
df = pd.DataFrame(out)
df.to_csv(OUT / "FRESH3_per_group.csv", index=False)
summ = df.groupby("method", sort=False)[["avg", "min", "sd"]].mean()
summ["nobody_hates"] = 1 - summ.sd
print(f"\n[3번 새 시험지 {len(groups)}그룹, 후보 평균 {rows.groupby('gid').movie_id.nunique().mean():.1f}편]\n",
      summ.round(4).to_string())
g = {m: df[df.method == m].sort_values("gid").reset_index(drop=True) for m in summ.index}
boot = {}
for a_, b_ in [("4차 (최종 후보)", "1차 Risk-Aware"), ("4차 (최종 후보)", "평균 최대"), ("1차 Risk-Aware", "평균 최대"),
               ("4차 (최종 후보)", "Least Misery"), ("4차 (최종 후보)", "한 사람 계정")]:
    boot[f"{a_} - {b_}"] = paired_bootstrap(g[a_], g[b_])
    print(f"  {a_} - {b_}: " + ", ".join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in boot[f'{a_} - {b_}'].items()))
d41 = boot["4차 (최종 후보) - 1차 Risk-Aware"]
adopt = d41["sd"]["diff"] < 0 and d41["avg"]["hi"] >= 0
print("\n약속에 따른 결론:", "4차 채택" if adopt else "1차 유지")
json.dump(dict(summary=summ.reset_index().to_dict("records"), bootstrap=boot, adopt_v4=bool(adopt), config=cfg4),
          open(OUT / "V4_FINAL_RESULTS.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
