"""[3차] 최종 채점: 2번 새 시험지(FRESH_HOLDOUT2, 500그룹, 1번 시험지와 사람이 안 겹침)로 1·2·3차를 한 번씩.

각 버전은 자기 고정 설정(results_v1_frozen / results_v2 / results_v3)을 그대로 쓴다.
극혐 판정 잣대는 각 버전이 고른 percentile 들로 모두 계산해서 같은 잣대로 비교한다.
실행: python 34_v3_final.py
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import ROOT, SEED, group_rows, load_movielens, movie_genre_table  # noqa: E402
from group import build_case, outcome, paired_bootstrap, summarize  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402
from risk_v2 import RiskContext, risk_proba_v2  # noqa: E402
from stage1_v3 import make_stage1  # noqa: E402

DIRS = {"v1": ROOT / "results_v1_frozen", "v2": ROOT / "results_v2", "v3": ROOT / "results_v3"}
OUT = ROOT / "results_v3"
r, groups = load_movielens()
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
mstats = movie_stats(trva)
truth = te.set_index(["user_id", "movie_id"]).rating_norm
V = {}
for name, d in DIRS.items():
    cfg = json.load(open(d / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
    with open(d / "risk_model.pkl", "rb") as f:
        rk = pickle.load(f)
    V[name] = dict(cfg=cfg, risk=rk, p=cfg["stage2"]["percentile"], lam=cfg["lam"], beta=cfg.get("beta", 0.0),
                   s1=make_stage1(cfg["stage1"], gt, seed=SEED, results_dir=d).fit(trva))
    V[name]["th"] = user_thresholds(trva, V[name]["p"])
    if rk.get("kind") == "indiv_v2":
        V[name]["ctx"] = RiskContext(trva, V[name]["th"].thr, genre_lookup, rk["alpha"], trva, rk["kappa"])
    print(name, "beta", V[name]["beta"], "lambda", V[name]["lam"], "p", V[name]["p"])
PS = sorted({v["p"] for v in V.values()})
TH = {p: user_thresholds(trva, p) for p in PS}

holdout = group_rows(pd.read_csv(OUT / "FRESH_HOLDOUT2_groups.csv"), "test")
cases = {name: [] for name in V}
for gid, users, movies in holdout:
    true = np.array([[truth[(u, m)] for u in users] for m in movies])
    for name, v in V.items():
        pred = np.column_stack([v["s1"].predict([u] * len(movies), movies) for u in users])
        thr, share = v["th"].thr.loc[users].to_numpy(), v["th"].share.loc[users].to_numpy()
        if v["risk"].get("kind") == "indiv_v2":
            rk = risk_proba_v2(v["risk"], pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                               np.array(movies), users, mstats, v["ctx"])
        else:
            rk = build_case(gid, users, movies, pred, true, thr, share, v["risk"], mstats)["risk"]
        cases[name].append(dict(group_id=gid, users=users, pred=pred, true=true, rk=rk))


def pick(c, name, method):
    p = c["pred"]
    if method == "Average":
        return int(np.argmax(p.mean(1)))
    if method == "LeastMisery":
        return int(np.argmax(p.min(1) + 1e-9 * p.mean(1)))
    if method == "Oracle":
        return int(np.argmax(c["true"].mean(1)))
    v = V[name]
    return int(np.argmax(p.mean(1) + v["beta"] * p.min(1) - v["lam"] * c["rk"]))


results = {}
for eval_p in PS:
    per, rows = {}, []
    for name in V:
        for m in ["Random", "Average", "LeastMisery", "Final", "Oracle"]:
            out = []
            for c in cases[name]:
                cc = dict(true=c["true"], thr=TH[eval_p].thr.loc[c["users"]].to_numpy())
                if m == "Random":
                    os_ = [outcome(cc, i) for i in range(len(c["true"]))]
                    out.append(dict(group_id=c["group_id"], **{k: float(np.mean([o[k] for o in os_])) for k in ("avg", "min", "sd")}))
                else:
                    out.append(dict(group_id=c["group_id"], **outcome(cc, pick(c, name, m))))
            df = pd.DataFrame(out)
            per[(name, m)] = df
            rows.append(dict(version=name, method=m, **summarize(df)))
    tab = pd.DataFrame(rows)
    comps = [("v3", "Final", "v1", "Final"), ("v3", "Final", "v1", "LeastMisery"), ("v3", "Final", "v3", "Average"),
             ("v1", "Final", "v1", "Average"), ("v1", "Final", "v1", "LeastMisery"), ("v3", "Final", "v2", "Final")]
    boot = {f"{a}{b} - {c}{d}": paired_bootstrap(per[(a, b)], per[(c, d)]) for a, b, c, d in comps}
    results[f"p{eval_p}"] = dict(table=tab.to_dict("records"), bootstrap=boot)
    print(f"\n[2번 새 시험지 {len(holdout)}그룹 | 극혐 잣대 하위 {eval_p}%]\n", tab.round(4).to_string(index=False))
    for k, v in boot.items():
        print(f"  {k}: " + ", ".join(f"{m} {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for m, d in v.items()))
    pd.concat([df.assign(version=k[0], method=k[1]) for k, df in per.items()]).to_csv(OUT / f"FRESH2_per_group_p{eval_p}.csv", index=False)

te_w = te[te.user_id.isin(trva.user_id) & te.movie_id.isin(trva.movie_id)]
results["test_individual_rmse_stars"] = {
    n: float(np.sqrt(np.mean((te_w.rating_norm.values - v["s1"].predict(te_w.user_id.values, te_w.movie_id.values)) ** 2)) * 4)
    for n, v in V.items()}
print("\n[Test 기간 개인 예측 RMSE, 별점]", {k: round(x, 4) for k, x in results["test_individual_rmse_stars"].items()})
with open(OUT / "V3_FINAL_RESULTS.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
