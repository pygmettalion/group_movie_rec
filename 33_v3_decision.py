"""[version 3] Choose the final scoring formula -> freeze v3.

Candidate formula:  Final = (mean of 4 predictions) + beta * (lowest of the 4 predictions) - lambda * (Strong Dislike risk)
  beta=0 is the same formula as v1/v2. A larger beta mixes in the strength of Least Misery (nobody is left out).

Selection data (both are practice data where 'choosing after seeing the answers is allowed'):
  (a) Validation, 649 groups (49 official + 600 practice)      : predictions of the model trained on Train
  (b) fresh holdout 1, 500 groups (already used for the v2 comparison) : predictions of the model trained on Train+Validation
      -> test-period groups are included to reduce why v2 lost (a lambda fitted to the Validation period was too strong in the test period).
Rule: the (beta, lambda) with the highest average balanced score over the two sets. Ties go to the smaller beta+lambda.
Final scoring uses only fresh holdout 2, which shares no person with the others.

Run: RESULTS_DIR=results_v3 python 33_v3_decision.py
"""
import itertools
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, ROOT, SEED, group_rows, load_json, load_movielens, movie_genre_table, save_json  # noqa: E402
from group import outcome, summarize  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402
from risk_v2 import RiskContext, risk_proba_v2  # noqa: E402
from stage1_v3 import make_stage1  # noqa: E402

BETAS = [0, 0.1, 0.25, 0.5, 1.0, 2.0]
LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

r, groups = load_movielens()
tr, va = r[r.split == "train"], r[r.split == "validation"]
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
s1sel, s2sel = load_json("stage1_selection_final.json"), load_json("stage2_selection.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
P = s2sel["percentile"]


def make_case(gid, pred, true, thr, rk):
    return dict(group_id=gid, pred=np.asarray(pred), true=np.asarray(true), thr=np.asarray(thr), risk=np.asarray(rk))


def risk_for(pred, users, movies, th, ms, ctx):
    thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
    return risk_proba_v2(risk, pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                         np.array(movies), users, ms, ctx), thr


# (a) Validation 649
preds = pd.read_csv(RESULTS / "val_stage1_predictions.csv").set_index(["user_id", "movie_id"]).pred
truth_v = va.set_index(["user_id", "movie_id"]).rating_norm
th_tr, ms_tr = user_thresholds(tr, P), movie_stats(tr)
ctx_tr = RiskContext(tr, th_tr.thr, genre_lookup, risk["alpha"], tr, risk["kappa"])
A = []
for gid, users, movies in group_rows(groups, "validation"):
    pred = np.array([[preds[(u, m)] for u in users] for m in movies])
    rk, thr = risk_for(pred, users, movies, th_tr, ms_tr, ctx_tr)
    A.append(make_case(gid, pred, [[truth_v[(u, m)] for u in users] for m in movies], thr, rk))
extra = pd.read_csv(RESULTS / "stage2_extra_val_groups.csv")
for gid, d in extra.groupby("group_id"):
    A.append(make_case(gid, d[["p1", "p2", "p3", "p4"]], d[["y1", "y2", "y3", "y4"]],
                       d[["thr1", "thr2", "thr3", "thr4"]].iloc[0], d.risk))

# (b) fresh holdout 1, 500 groups (trained on Train+Validation)
s1 = make_stage1(s1sel, gt, seed=SEED, results_dir=RESULTS).fit(trva)
th_tv, ms_tv = user_thresholds(trva, P), movie_stats(trva)
ctx_tv = RiskContext(trva, th_tv.thr, genre_lookup, risk["alpha"], trva, risk["kappa"])
truth_t = te.set_index(["user_id", "movie_id"]).rating_norm
B = []
for gid, users, movies in group_rows(pd.read_csv(ROOT / "results_v2" / "FRESH_HOLDOUT_groups.csv"), "test"):
    pred = np.column_stack([s1.predict([u] * len(movies), movies) for u in users])
    rk, thr = risk_for(pred, users, movies, th_tv, ms_tv, ctx_tv)
    B.append(make_case(gid, pred, [[truth_t[(u, m)] for u in users] for m in movies], thr, rk))
print(f"선택용 그룹: Validation {len(A)} + 1번 시험지 {len(B)}")


def run(cases, beta, lam):
    rows = []
    for c in cases:
        s = c["pred"].mean(1) + beta * c["pred"].min(1) - lam * c["risk"]
        rows.append(outcome(c, int(np.argmax(s))))
    return summarize(pd.DataFrame(rows))


grid = []
for beta, lam in itertools.product(BETAS, LAMBDAS):
    a, b = run(A, beta, lam), run(B, beta, lam)
    grid.append(dict(beta=beta, lam=lam, score=(a["balanced_score"] + b["balanced_score"]) / 2,
                     **{f"val_{k}": v for k, v in a.items()}, **{f"h1_{k}": v for k, v in b.items()}))
g = pd.DataFrame(grid)
g.to_csv(RESULTS / "v3_decision_grid.csv", index=False)
top = g[g.score >= g.score.max() - 1e-12].assign(s=lambda d: d.beta + d.lam).sort_values(["s", "beta"]).iloc[0]
print(g.sort_values("score", ascending=False).head(10).round(4).to_string(index=False))
ref = g[(g.beta == 0)].sort_values("score", ascending=False).iloc[0]
print(f"\n선택: beta={top.beta}, lambda={top.lam}  (beta=0 중 최고: lambda={ref.lam}, score {ref.score:.4f})")
save_json(dict(version="v3", frozen_note="3차 고정. 2번 새 시험지(FRESH_HOLDOUT2)로 1·2·3차를 한 번씩만 채점.",
               stage1=s1sel, stage2=s2sel, beta=float(top.beta), lam=float(top.lam),
               selection_score=float(top.score), rule="Validation649 와 1번시험지500 의 balanced 평균 최대, 동점이면 beta+lambda 작은 쪽"),
          "FINAL_FROZEN_CONFIG.json")
print("3차 모델 고정 완료")
