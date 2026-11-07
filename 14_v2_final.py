"""[version 2] Final scoring.

(a) fresh holdout (FRESH_HOLDOUT, 500 sealed groups): v1 frozen model vs v2 frozen model -> fair comparison (main result)
(b) the existing Internal / External / Beliefs: v2 model scores (reference only; this model was built after seeing v1 results, so it is reported as a 'secondary result')

The Strong Dislike threshold (percentile) may differ between v1 and v2, so the Strong Dislike rate is computed under both thresholds and compared on the same yardstick.
Run: python 14_v2_final.py   (reads results_v1_frozen and results_v2)
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import (ROOT, SEED, group_rows, load_beliefs, load_movielens, load_movietweetings,  # noqa: E402
                    movie_genre_table, rmse)
from group import build_case, evaluate, paired_bootstrap, summarize  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402
from risk_v2 import RiskContext, risk_proba_v2  # noqa: E402

OUT = ROOT / "results_v2"
V = {}
for name, d in [("v1", ROOT / "results_v1_frozen"), ("v2", OUT)]:
    cfg = json.load(open(d / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
    with open(d / "risk_model.pkl", "rb") as f:
        V[name] = dict(cfg=cfg, risk=pickle.load(f), p=cfg["stage2"]["percentile"], lam=cfg["lam"])
print({k: (v["cfg"]["stage1"]["structure"], v["p"], v["lam"]) for k, v in V.items()})

r, groups = load_movielens()
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
mstats = movie_stats(trva)
truth = te.set_index(["user_id", "movie_id"]).rating_norm
for name in V:
    V[name]["s1"] = Stage1Predictor(V[name]["cfg"]["stage1"], gt, seed=SEED).fit(trva)
PS = sorted({V["v1"]["p"], V["v2"]["p"]})
TH = {p: user_thresholds(trva, p) for p in PS}
if V["v2"]["risk"].get("kind") == "indiv_v2":
    V["v2"]["ctx"] = RiskContext(trva, TH[V["v2"]["p"]].thr, genre_lookup, V["v2"]["risk"]["alpha"], trva,
                                 V["v2"]["risk"]["kappa"])


def risk_of(name, pred, users, movies, th):
    thr, share = th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy()
    T, S = np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1))
    rk = V[name]["risk"]
    if rk.get("kind") == "indiv_v2":
        return risk_proba_v2(rk, pred, T, S, np.array(movies), users, mstats, V[name]["ctx"])
    return build_case(0, users, movies, pred, pred, thr, share, rk, mstats)["risk"]


def ml_cases(name, group_list, eval_p):
    """eval_p: the percentile used to judge Strong Dislike (the scoring yardstick). Inside each model, risk is computed with that version's own percentile."""
    out = []
    for gid, users, movies in group_list:
        pred = np.column_stack([V[name]["s1"].predict([u] * len(movies), movies) for u in users])
        true = np.array([[truth[(u, m)] for u in users] for m in movies])
        rk = risk_of(name, pred, users, movies, TH[V[name]["p"]])
        out.append(dict(group_id=gid, users=users, movies=movies, pred=pred, true=true,
                        thr=TH[eval_p].thr.loc[users].to_numpy(), exp_sat=pred.mean(1), risk=rk))
    return out


def report(cases_by_version, tag):
    rows, per = [], {}
    for name, cs in cases_by_version.items():
        for m in ["Random", "Average", "LeastMisery", "RiskAware", "Oracle"]:
            lam = V[name]["lam"] if m == "RiskAware" else 0.0
            df = evaluate(cs, m, lam).sort_values("group_id").reset_index(drop=True)
            per[(name, m)] = df
            rows.append(dict(version=name, method=m, **summarize(df)))
    tab = pd.DataFrame(rows)
    return tab, per


# ================================================================ (a) fresh holdout
fresh = pd.read_csv(OUT / "FRESH_HOLDOUT_groups.csv")
fresh_list = group_rows(fresh, "test")
results = {}
for eval_p in PS:
    cs = {name: ml_cases(name, fresh_list, eval_p) for name in V}
    tab, per = report(cs, f"fresh_p{eval_p}")
    tab.insert(0, "dislike_percentile", eval_p)
    boot = {
        "v2 RiskAware - v1 RiskAware": paired_bootstrap(per[("v2", "RiskAware")], per[("v1", "RiskAware")]),
        "v2 RiskAware - v2 Average": paired_bootstrap(per[("v2", "RiskAware")], per[("v2", "Average")]),
        "v1 RiskAware - v1 Average": paired_bootstrap(per[("v1", "RiskAware")], per[("v1", "Average")]),
        "v2 Average - v1 Average": paired_bootstrap(per[("v2", "Average")], per[("v1", "Average")]),
    }
    results[f"fresh_p{eval_p}"] = dict(table=tab.to_dict("records"), bootstrap=boot)
    print(f"\n[새 시험지 {len(fresh_list)}그룹 | Strong Dislike 잣대 = 하위 {eval_p}%]\n", tab.round(4).to_string(index=False))
    for k, v in boot.items():
        print(f"  {k}: " + ", ".join(f"{m} {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for m, d in v.items()))
    pd.concat([df.assign(version=k[0]) for k, df in per.items()]).to_csv(OUT / f"FRESH_per_group_p{eval_p}.csv", index=False)

# individual prediction accuracy (all Test-period ratings)
te_w = te[te.user_id.isin(trva.user_id) & te.movie_id.isin(trva.movie_id)]
indiv = {name: rmse(te_w.rating_norm, V[name]["s1"].predict(te_w.user_id.values, te_w.movie_id.values)) * 4 for name in V}
print("\n[Test 기간 개인 예측 RMSE, 별점]", {k: round(v, 4) for k, v in indiv.items()})
results["test_individual_rmse_stars"] = indiv

# ================================================================ (b) existing test sets (v2, reference only)
p2 = V["v2"]["p"]
int_tab, int_per = report({"v2": ml_cases("v2", group_rows(groups, "test"), p2)}, "internal")
print("\n[참고] 기존 Internal 49그룹 (2차)\n", int_tab.round(4).to_string(index=False))
results["internal_v2_reference"] = int_tab.to_dict("records")

mt, mt_groups = load_movietweetings()
hist, mt_test = mt[mt.split == "history"], mt[mt.split == "test"]
gt_mt = {int(k) if k.isdigit() else k: v for k, v in movie_genre_table(mt).items()}
th_mt = user_thresholds(hist, p2)
truth_mt = mt_test.assign(mid=mt_test.movie_id.astype(int)).set_index(["user_id", "mid"]).rating_norm
hb = {u: d for u, d in hist.groupby("user_id")}
ctx_mt = RiskContext(hist, th_mt.thr, genre_lookup, V["v2"]["risk"]["alpha"], trva, V["v2"]["risk"]["kappa"])
mt_cases = []
for gid, users, movies in group_rows(mt_groups, "test"):
    pred = np.column_stack([V["v2"]["s1"].predict_new_user([int(m) if m.isdigit() else m for m in hb[u].movie_id],
                                                           hb[u].rating_norm.values, movies, genre_table=gt_mt)[0]
                            for u in users])
    true = np.array([[truth_mt[(u, m)] for u in users] for m in movies])
    thr, share = th_mt.thr.loc[users].to_numpy(), th_mt.share.loc[users].to_numpy()
    rk = risk_proba_v2(V["v2"]["risk"], pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                       np.array(movies), users, mstats, ctx_mt)
    mt_cases.append(dict(group_id=gid, users=users, movies=movies, pred=pred, true=true, thr=thr,
                         exp_sat=pred.mean(1), risk=rk))
ext_tab, _ = report({"v2": mt_cases}, "external")
print("\n[참고] 기존 External 44그룹 (2차)\n", ext_tab.round(4).to_string(index=False))
results["external_v2_reference"] = ext_tab.to_dict("records")

b_test, b_hist = load_beliefs()
gt_b = movie_genre_table(b_hist, b_test)
b_hist = b_hist.sort_values(["user_id", "timestamp"])
hg = {u: (d.timestamp.to_numpy(), d.movie_id.to_numpy(), d.rating_norm.to_numpy()) for u, d in b_hist.groupby("user_id")}
rows = []
for (u, ts), d in b_test.groupby(["user_id", "timestamp"]):
    hts, hm, hy = hg.get(u, (np.array([]), np.array([]), np.array([])))
    cut = np.searchsorted(hts, ts, side="left")
    pred, known = V["v2"]["s1"].predict_new_user(hm[:cut], hy[:cut], d.movie_id.to_numpy(), genre_table=gt_b,
                                                 allow_content_fallback=True)
    rows.append(pd.DataFrame({"y": d.rating_norm.values, "pred": pred, "known": known,
                              "umean": hy[:cut].mean() if cut else np.nan}))
bel = pd.concat(rows)
bel = bel[bel.umean.notna()]
btab = [dict(subset=n, n=int(k.sum()), model_rmse_stars=rmse(bel.y[k], bel.pred[k]) * 4.5,
             user_mean_rmse_stars=rmse(bel.y[k], bel.umean[k]) * 4.5)
        for n, k in [("전체", np.ones(len(bel), bool)), ("MF가 아는 영화", bel.known.to_numpy()),
                     ("새 영화(Content)", ~bel.known.to_numpy())]]
print("\n[참고] Beliefs (2차)\n", pd.DataFrame(btab).round(4).to_string(index=False))
results["beliefs_v2_reference"] = btab
with open(OUT / "V2_FINAL_RESULTS.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=float)
print("\n저장 -> results_v2/V2_FINAL_RESULTS.json")
