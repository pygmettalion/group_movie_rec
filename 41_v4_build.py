"""[4차] 팀 목표("아무도 극혐하지 않는 선택")를 직접 노리도록 '고르는 방식'을 다시 만든다.

1~3차와 다른 점
 (1) 고르는 기준을 팀 목표에 맞춤 (결과 보기 전에 고정):
       극혐 발생률을 최소화한다. 단, 평균 만족도가 '평균 최대' 방식보다 0.005 넘게 떨어지면 탈락.
       동점이면 최저 만족도가 높은 것 -> 평균 만족도가 높은 것 -> 더 단순한 것.
 (2) 거부권(veto) 규칙 후보 추가: 극혐 위험이 tau 를 넘는 영화는 빼고, 남은 영화 중 평균 1등.
     (모두 빠지면 위험이 가장 낮은 영화)
 (3) 시험 기간에 맞춘 위험 모델 후보 추가: 1·2번 시험지 사용자들의 '시험 기간 실제 평점'으로 학습.
     1번 시험지 그룹의 위험은 2번 시험지 사람들로만 학습한 모델로, 2번은 1번 사람들로만 학습한 모델로 계산 (교차 적합).
고르는 데이터: 이미 사용한 1·2번 시험지 1,000그룹 (시험 기간, 실제 서비스 상황과 같은 시점).
최종 채점: 아무도 본 적 없는 3번 시험지(06 으로 봉인)에서 1차 vs 4차 한 번.

실행: python 41_v4_build.py
"""
import itertools
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, "src")
from common import ROOT, SEED, group_rows, load_movielens, movie_genre_table  # noqa: E402
from risk import indiv_features, movie_stats, user_thresholds  # noqa: E402
from risk_v2 import INDIV_V2_SETS, RiskContext, indiv_features_v2  # noqa: E402
from stage1_v3 import make_stage1  # noqa: E402

OUT = ROOT / "results_v4"
OUT.mkdir(exist_ok=True)
P, ALPHA, KAPPA, TOL = 5, 3.0, 50.0, 0.005
r, _ = load_movielens()
trva, te = r[r.split.isin(["train", "validation"])], r[r.split == "test"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
mstats = movie_stats(trva)
th = user_thresholds(trva, P)
ctx = RiskContext(trva, th.thr, genre_lookup, ALPHA, trva, KAPPA)

H = {1: pd.read_csv(ROOT / "results_v2" / "FRESH_HOLDOUT_groups.csv"),
     2: pd.read_csv(ROOT / "results_v3" / "FRESH_HOLDOUT2_groups.csv")}
users_of = {h: set(df[[f"user_id_{i}" for i in range(1, 5)]].values.ravel()) for h, df in H.items()}
rows = te[te.user_id.isin(users_of[1] | users_of[2]) & te.movie_id.isin(trva.movie_id)].copy()
rows["h"] = np.where(rows.user_id.isin(users_of[1]), 1, 2)
print(f"시험 기간 평점 (1·2번 시험지 사용자): {len(rows):,}건")

# ---------------------------------------------------------------- 개인 예측: 1차 Stage 1 / 3차 Stage 1 (Train+Validation 학습)
cfg1 = json.load(open(ROOT / "results_v1_frozen" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
cfg3 = json.load(open(ROOT / "results_v3" / "FINAL_FROZEN_CONFIG.json", encoding="utf-8"))
S = {"S1": make_stage1(cfg1["stage1"], gt, SEED).fit(trva),
     "S3": make_stage1(cfg3["stage1"], gt, SEED, results_dir=ROOT / "results_v3").fit(trva)}
for k, s in S.items():
    rows[f"pred_{k}"] = s.predict(rows.user_id.values, rows.movie_id.values)
rows["thr"] = rows.user_id.map(th.thr)
rows["share"] = rows.user_id.map(th.share)
rows["dis"] = (rows.rating_norm <= rows.thr).astype(int)
print("시험 기간 개인 극혐 비율:", round(rows.dis.mean(), 4))

# ---------------------------------------------------------------- 위험 모델 후보
with open(ROOT / "results_v1_frozen" / "risk_model.pkl", "rb") as f:
    R1 = pickle.load(f)
with open(ROOT / "results_v3" / "risk_model.pkl", "rb") as f:
    R3 = pickle.load(f)
X1 = indiv_features(rows.pred_S1.values, rows.thr.values, rows.share.values, rows.movie_id.values, mstats)
rows["p_RV1"] = R1["model"].predict_proba(X1[R1["features"]])[:, 1]          # 1차 위험 모델 (Validation 기간 학습)
X3 = indiv_features_v2(rows.pred_S3.values, rows.thr.values, rows.share.values, rows.movie_id.values,
                       rows.user_id.values, mstats, ctx)
rows["p_RV3"] = R3["model"].predict_proba(X3[R3["features"]])[:, 1]          # 3차 위험 모델 (Validation 기간 학습)

feats = INDIV_V2_SETS["I_gd"]
CLF = {"HGB(lr=0.1,leaves=7)": dict(learning_rate=0.1, max_leaf_nodes=7),
       "HGB(lr=0.03,leaves=31)": dict(learning_rate=0.03, max_leaf_nodes=31)}
rt_report = []
for sname in ["S1", "S3"]:
    X = indiv_features_v2(rows[f"pred_{sname}"].values, rows.thr.values, rows.share.values, rows.movie_id.values,
                          rows.user_id.values, mstats, ctx)
    best = None
    for cname, kw in CLF.items():
        oof = np.zeros(len(rows))
        for h in (1, 2):
            a, b = (rows.h != h).to_numpy(), (rows.h == h).to_numpy()
            m = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0, random_state=SEED, **kw)
            m.fit(X.loc[a, feats], rows.dis.values[a])
            oof[b] = m.predict_proba(X.loc[b, feats])[:, 1]
        ll = log_loss(rows.dis, np.clip(oof, 1e-6, 1 - 1e-6))
        rt_report.append(dict(stage1=sname, classifier=cname, logloss=ll, auc=roc_auc_score(rows.dis, oof)))
        if best is None or ll < best[0]:
            best = (ll, cname, oof)
    rows[f"p_RT_{sname}"] = best[2]
    # 같은 행에서 Validation 학습 위험 모델과 비교 (참고)
    base = rows["p_RV1" if sname == "S1" else "p_RV3"]
    rt_report.append(dict(stage1=sname, classifier="Validation 학습 모델(참고)",
                          logloss=log_loss(rows.dis, np.clip(base, 1e-6, 1 - 1e-6)), auc=roc_auc_score(rows.dis, base)))
    print(f"[{sname}] 시험 기간 학습 위험 모델 {best[1]}: logloss {best[0]:.4f} / Validation 학습 모델 "
          f"{rt_report[-1]['logloss']:.4f}")
    rows[f"clf_RT_{sname}"] = best[1]
pd.DataFrame(rt_report).to_csv(OUT / "v4_risk_models.csv", index=False)

# ---------------------------------------------------------------- 그룹 단위로 묶기
key = rows.set_index(["user_id", "movie_id"])
COMBOS = {"S1+RV1": ("pred_S1", "p_RV1"), "S1+RT": ("pred_S1", "p_RT_S1"),
          "S3+RV3": ("pred_S3", "p_RV3"), "S3+RT": ("pred_S3", "p_RT_S3")}
cases = []
for h, df in H.items():
    for gid, users, movies in group_rows(df, "test"):
        idx = [(u, m) for m in movies for u in users]
        sub = key.loc[idx]
        c = dict(h=h, gid=gid, true=sub.rating_norm.to_numpy().reshape(len(movies), 4),
                 thr=th.thr.loc[users].to_numpy())
        for name, (pc, rc) in COMBOS.items():
            c[f"pred_{name}"] = sub[pc].to_numpy().reshape(len(movies), 4)
            c[f"risk_{name}"] = 1 - np.prod(1 - sub[rc].to_numpy().reshape(len(movies), 4), axis=1)
        cases.append(c)


def choose(c, combo, rule, a, b):
    p, rk = c[f"pred_{combo}"], c[f"risk_{combo}"]
    mean = p.mean(1)
    if rule == "AVG":
        return int(np.argmax(mean))
    if rule == "LIN":
        return int(np.argmax(mean - a * rk))
    if rule == "LINMIN":
        return int(np.argmax(mean + b * p.min(1) - a * rk))
    if rule == "VETO":
        ok = rk <= a
        return int(np.argmax(np.where(ok, mean, -np.inf))) if ok.any() else int(np.argmin(rk))
    raise ValueError(rule)


def evaluate(cs, combo, rule, a=0.0, b=0.0):
    o = []
    for c in cs:
        i = choose(c, combo, rule, a, b)
        t = c["true"][i]
        o.append(dict(gid=(c["h"], c["gid"]), avg=t.mean(), min=t.min(), sd=float((t <= c["thr"]).any())))
    return pd.DataFrame(o)


LAMS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
TAUS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
BETAS = [0.25, 0.5, 1.0]
CANDS = []
for combo in COMBOS:
    CANDS += [(combo, "LIN", l, 0.0) for l in LAMS]
    CANDS += [(combo, "VETO", t, 0.0) for t in TAUS]
    CANDS += [(combo, "LINMIN", l, bb) for l, bb in itertools.product(LAMS, BETAS)]
COMPLEXITY = {"LIN": 1, "VETO": 1, "LINMIN": 2}


def select(cs):
    """팀 목표 기준으로 최고 후보 고르기. 반환: (후보, 결과표)"""
    res = []
    for combo, rule, a, b in CANDS:
        e = evaluate(cs, combo, rule, a, b)
        base = evaluate(cs, combo, "AVG")
        res.append(dict(combo=combo, rule=rule, a=a, b=b, avg=e.avg.mean(), min=e["min"].mean(), sd=e.sd.mean(),
                        base_avg=base.avg.mean(), ok=e.avg.mean() >= base.avg.mean() - TOL,
                        complexity=COMPLEXITY[rule] + (0 if combo.endswith("RV1") or combo.endswith("RV3") else 1)))
    t = pd.DataFrame(res)
    ok = t[t.ok].sort_values(["sd", "min", "avg", "complexity"], ascending=[True, False, False, True])
    return ok.iloc[0], t


# ---------------------------------------------------------------- (참고) 정직한 성능 추정: 한쪽 시험지로 고르고 다른 쪽으로 채점
nested = []
for pick_h, eval_h in [(1, 2), (2, 1)]:
    best, _ = select([c for c in cases if c["h"] == pick_h])
    ev = [c for c in cases if c["h"] == eval_h]
    v4 = evaluate(ev, best.combo, best.rule, best.a, best.b)
    v1 = evaluate(ev, "S1+RV1", "LIN", 0.75)
    avg = evaluate(ev, "S1+RV1", "AVG")
    nested.append(dict(pick_on=pick_h, eval_on=eval_h, chosen=f"{best.combo} {best.rule} a={best.a} b={best.b}",
                       v4_avg=v4.avg.mean(), v4_min=v4["min"].mean(), v4_sd=v4.sd.mean(),
                       v1_avg=v1.avg.mean(), v1_min=v1["min"].mean(), v1_sd=v1.sd.mean(),
                       avg_avg=avg.avg.mean(), avg_sd=avg.sd.mean()))
    print(f"[교차 추정] {pick_h}번으로 고름 -> {eval_h}번 채점: {nested[-1]['chosen']} | 극혐 4차 {v4.sd.mean():.3f} / "
          f"1차 {v1.sd.mean():.3f} / 평균최대 {avg.sd.mean():.3f} | 평균만족 4차 {v4.avg.mean():.4f} / 1차 {v1.avg.mean():.4f}")
pd.DataFrame(nested).to_csv(OUT / "v4_nested_estimate.csv", index=False)

# ---------------------------------------------------------------- 최종 선택 (1·2번 전체) -> 고정
best, table = select(cases)
table.to_csv(OUT / "v4_selection_grid.csv", index=False)
print(table[table.ok].sort_values(["sd", "min"], ascending=[True, False]).head(10).round(4).to_string(index=False))
print("\n4차 선택:", best.to_dict())

# 시험 기간 위험 모델을 쓰면 1·2번 사용자 전체로 다시 학습해서 고정
combo = best.combo
sname = combo.split("+")[0]
final = dict(version="v4", combo=combo, rule=best.rule, a=float(best.a), b=float(best.b), percentile=P,
             tolerance=TOL, stage1_from="results_v1_frozen" if sname == "S1" else "results_v3",
             selection=dict(avg=best.avg, min=best["min"], sd=best.sd, base_avg=best.base_avg),
             rule_text="극혐률 최소 (평균 만족도 손실 0.005 이내), 동점이면 최저만족↑ -> 평균만족↑ -> 단순")
if combo.endswith("RT"):
    X = indiv_features_v2(rows[f"pred_{sname}"].values, rows.thr.values, rows.share.values, rows.movie_id.values,
                          rows.user_id.values, mstats, ctx)
    kw = CLF[rows[f"clf_RT_{sname}"].iloc[0]]
    m = HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0, random_state=SEED, **kw).fit(X[feats], rows.dis)
    risk = dict(model=m, features=feats, kind="indiv_v2", alpha=ALPHA, kappa=KAPPA, percentile=P)
else:
    risk = R1 if combo.endswith("RV1") else R3
with open(OUT / "risk_model.pkl", "wb") as f:
    pickle.dump(risk, f)
json.dump(final, open(OUT / "FINAL_FROZEN_CONFIG.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
print("4차 고정 완료")
