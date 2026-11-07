"""[2차 버전] lambda 다시 고르기 -> 2차 모델 고정. 1차와 같은 규칙, 같은 종류의 Validation 그룹.

실행: RESULTS_DIR=results_v2 python 13_v2_lambda.py
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, group_rows, load_json, load_movielens, save_json  # noqa: E402
from group import LAMBDA_GRID, evaluate, summarize  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402
from risk_v2 import RiskContext, risk_proba_v2  # noqa: E402

r, groups = load_movielens()
tr, va = r[r.split == "train"], r[r.split == "validation"]
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
s1sel, s2sel = load_json("stage1_selection_final.json"), load_json("stage2_selection.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
p = s2sel["percentile"]
preds = pd.read_csv(RESULTS / "val_stage1_predictions.csv").set_index(["user_id", "movie_id"]).pred
truth = va.set_index(["user_id", "movie_id"]).rating_norm
th = user_thresholds(tr, p)
mstats = movie_stats(tr)
ctx = RiskContext(tr, th.thr, genre_lookup, risk["alpha"], tr, risk["kappa"])


def case(gid, users, movies, pred, true, thr, rk):
    return dict(group_id=gid, users=list(users), movies=list(movies), pred=pred, true=true, thr=np.asarray(thr),
                exp_sat=pred.mean(1), risk=rk)


cases = []
for gid, users, movies in group_rows(groups, "validation"):
    pred = np.array([[preds[(u, m)] for u in users] for m in movies])
    true = np.array([[truth[(u, m)] for u in users] for m in movies])
    thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
    rk = risk_proba_v2(risk, pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                       np.array(movies), users, mstats, ctx)
    cases.append(case(gid, users, movies, pred, true, thr, rk))
extra = pd.read_csv(RESULTS / "stage2_extra_val_groups.csv")
for gid, d in extra.groupby("group_id"):
    cases.append(case(gid, d.users.iloc[0].split("|"), d.movie_id.tolist(), d[["p1", "p2", "p3", "p4"]].to_numpy(),
                      d[["y1", "y2", "y3", "y4"]].to_numpy(), d[["thr1", "thr2", "thr3", "thr4"]].iloc[0].to_numpy(),
                      d.risk.to_numpy()))
official = cases[:49]
print(f"lambda 선택용 그룹 {len(cases)}개 (공식 {len(official)} + 연습 {len(cases) - len(official)})")

rows = [dict(lam=lam, **summarize(evaluate(cases, "RiskAware", lam))) for lam in LAMBDA_GRID]
# 최적값이 격자 끝(5.0)이면 더 큰 값까지 넓혀서 다시 확인 (다른 숫자들과 같은 원칙)
grid = list(LAMBDA_GRID)
while max(rows, key=lambda x: (x["balanced_score"], -x["lam"]))["lam"] == grid[-1] and grid[-1] < 1000:
    for lam in [grid[-1] * 1.5, grid[-1] * 2, grid[-1] * 4]:
        rows.append(dict(lam=lam, **summarize(evaluate(cases, "RiskAware", lam))))
    grid = sorted(x["lam"] for x in rows)
lam_df = pd.DataFrame(rows).sort_values("lam")
lam_df.to_csv(RESULTS / "stage3_lambda_search.csv", index=False)
print(lam_df.round(4).to_string(index=False))
lam_star = float(lam_df[lam_df.balanced_score >= lam_df.balanced_score.max() - 1e-12].lam.min())
print("선택 lambda =", lam_star)
base = [dict(method=m, **summarize(evaluate(cases, m))) for m in ["Random", "Average", "LeastMisery", "Oracle"]]
base.append(dict(method=f"RiskAware(lambda={lam_star})", **summarize(evaluate(cases, "RiskAware", lam_star))))
pd.DataFrame(base).to_csv(RESULTS / "stage3_validation_methods.csv", index=False)
print(pd.DataFrame(base).round(4).to_string(index=False))
save_json(dict(version="v2", frozen_note="2차 버전 고정. 새 시험지(FRESH_HOLDOUT)로 1차와 함께 한 번만 채점.",
               stage1=s1sel, stage2=s2sel, lam=lam_star, n_lambda_groups=len(cases),
               lambda_rule="balanced_score=(평균만족+최저만족+(1-Strong Dislike 비율))/3 최대, 동점이면 작은 lambda"),
          "FINAL_FROZEN_CONFIG.json")
print("2차 모델 고정 완료")
