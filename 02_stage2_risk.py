"""Stage 2: Strong Dislike 기준(percentile) + Group Risk 분류기 고르기.

쉽게 말하면
  "이 영화를 4명이 보면, 그중 누군가 '최악이야!' 할 확률" 을 맞히는 모델을 만든다.

학습 데이터를 어떻게 만드나?
  - 공식 평가 그룹은 49개뿐이라 분류기를 학습시키기엔 너무 적다.
  - 그래서 Validation 기간에 '같은 영화를 본 사람 4명'을 무작위로 묶어 연습용 가짜 그룹(pseudo-group)을 만든다.
  - 공식 평가 그룹 사용자 36명은 여기서 완전히 제외한다 -> 나중에 lambda 를 고를 때 쓰는 데이터와 겹치지 않음.
  - 개인 선호 점수는 Train 만으로 학습한 Stage 1 모델이 낸 값(= 그 영화 평점은 모르는 상태의 예측)을 쓴다.

무엇을 비교하나?
  - Strong Dislike percentile 후보: 5, 10, 15, 20, 25
  - 분류기: Logistic Regression / HistGradientBoosting / RandomForest (각각 몇 가지 설정)
  - 피처 조합: F1(예측점수만) / F2(+개인 불호 기준선과의 거리) / F3(+영화 과거 평점 통계)
  - 채점: 영화 단위로 나눈 5-fold 교차검증 (같은 영화가 학습/채점에 동시에 들어가지 않게)

선택 규칙 (결과를 보기 전에 정해 둠)
  1) 각 percentile 안에서: 교차검증 log loss 가 가장 낮은 (분류기, 설정, 피처조합)
     -> 확률값 자체가 최종 점수에 들어가므로 '확률이 잘 맞는지'를 보는 log loss 를 쓴다.
  2) percentile 끼리: 위 모델의 ROC-AUC(위험한 영화를 얼마나 잘 골라내는지, 기준선 비율과 무관)가 가장 높은 값.
     AUC 차이가 0.005 이내면 더 작은 percentile(더 '강한' 싫음)을 고른다.
"""
import itertools
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_json, load_movielens, movie_genre_table, save_json  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import (FEATURE_SETS, INDIV_FEATURES, PERCENTILES, group_features, indiv_features, movie_stats,  # noqa: E402
                  user_thresholds)

MIN_HISTORY = 10          # 그룹 사용자와 같은 조건(과거 평점 10개 이상)
GROUPS_PER_MOVIE = 20     # 영화 하나당 가짜 그룹 최대 개수 (인기 영화가 데이터를 독차지하지 않게)

r, groups = load_movielens()
tr = r[r.split == "train"]
va = r[r.split == "validation"]
gt = movie_genre_table(r)
sel = load_json("stage1_selection_final.json")

# ---------------------------------------------------------------- Stage 1 (Train 만으로 학습) 예측
print("Stage 1 모델을 Train 으로 학습 ->", sel["structure"])
s1 = Stage1Predictor(sel, gt, seed=SEED).fit(tr)
counts = tr.groupby("user_id").size()
eval_users = set(groups[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())

va_w = va[va.user_id.isin(counts[counts >= MIN_HISTORY].index) & va.movie_id.isin(tr.movie_id)].copy()
va_w["pred"] = s1.predict(va_w.user_id.values, va_w.movie_id.values)
va_w[["user_id", "movie_id", "rating_norm", "pred"]].to_csv(RESULTS / "val_stage1_predictions.csv", index=False)

# ---------------------------------------------------------------- 가짜 그룹 만들기
pool = va_w[~va_w.user_id.isin(eval_users)]
rng = np.random.default_rng(SEED)
cols = ["movie_id", "u1", "u2", "u3", "u4", "y1", "y2", "y3", "y4", "p1", "p2", "p3", "p4"]


def make_pseudo(pool_df, rng):
    """같은 영화를 Validation 기간에 본 사람 4명을 무작위로 묶는다 (영화당 최대 GROUPS_PER_MOVIE 개)."""
    rows = []
    for movie, d in pool_df.groupby("movie_id"):
        if len(d) < 4:
            continue
        seen, tries = set(), 0
        while len(seen) < GROUPS_PER_MOVIE and tries < GROUPS_PER_MOVIE * 5:
            tries += 1
            pick = tuple(sorted(rng.choice(len(d), 4, replace=False)))
            if pick in seen:
                continue
            seen.add(pick)
            sub = d.iloc[list(pick)]
            rows.append((movie, *sub.user_id.values, *sub.rating_norm.values, *sub.pred.values))
    return pd.DataFrame(rows, columns=cols)


pg = make_pseudo(pool, rng)
print(f"가짜 그룹 {len(pg):,}개 (영화 {pg.movie_id.nunique():,}편, 사용자 {pool.user_id.nunique()}명)")

UC, YC, PC = ["u1", "u2", "u3", "u4"], ["y1", "y2", "y3", "y4"], ["p1", "p2", "p3", "p4"]
mstats = movie_stats(tr)


def build(pg, p):
    th = user_thresholds(tr, p)
    thr = np.column_stack([pg[c].map(th.thr).to_numpy() for c in UC])
    share = np.column_stack([pg[c].map(th.share).to_numpy() for c in UC])
    X = group_features(pg[PC].to_numpy(), thr, share, pg.movie_id.to_numpy(), mstats)
    y_true = pg[YC].to_numpy()
    label = (y_true <= thr).any(1).astype(int)
    indiv_rate = float((y_true <= thr).mean())
    return X, label, indiv_rate


CLASSIFIERS = {
    **{f"LogReg(C={C})": (lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))
       for C in [0.01, 0.1, 1, 10]},
    **{f"HGB(lr={lr},leaves={lv})": (lambda lr=lr, lv=lv: HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=lv, max_iter=200, l2_regularization=1.0, random_state=SEED))
       for lr, lv in itertools.product([0.03, 0.1], [7, 31])},
    **{f"RF(leaf={lf})": (lambda lf=lf: RandomForestClassifier(
        n_estimators=150, min_samples_leaf=lf, max_features="sqrt", n_jobs=-1, random_state=SEED))
       for lf in [30, 150]},
}

# 영화 단위 5-fold: 같은 영화는 항상 같은 fold (그룹 분류기와 개인 분류기가 똑같은 나눔을 쓰게)
movies_perm = rng.permutation(pool.movie_id.unique())
movie_fold = {m: i % 5 for i, m in enumerate(movies_perm)}
pg_fold = pg.movie_id.map(movie_fold).to_numpy()
pool = pool.assign(fold=pool.movie_id.map(movie_fold))

INDIV_CLASSIFIERS = {k: v for k, v in CLASSIFIERS.items() if k in ("LogReg(C=1)", "HGB(lr=0.1,leaves=7)", "HGB(lr=0.03,leaves=31)")}


def build_indiv(p, pool=pool, pg=pg):
    th = user_thresholds(tr, p)
    Xi = indiv_features(pool.pred.to_numpy(), pool.user_id.map(th.thr).to_numpy(),
                        pool.user_id.map(th.share).to_numpy(), pool.movie_id.to_numpy(), mstats)
    yi = (pool.rating_norm.to_numpy() <= pool.user_id.map(th.thr).to_numpy()).astype(int)
    # 가짜 그룹 4명 각각의 개인 피처 (위험확률을 4명 곱으로 합치기 위해)
    Xg = [indiv_features(pg[pc].to_numpy(), pg[uc].map(th.thr).to_numpy(), pg[uc].map(th.share).to_numpy(),
                         pg.movie_id.to_numpy(), mstats) for pc, uc in zip(PC, UC)]
    return Xi, yi, Xg


search, per_p = [], {}
for p in PERCENTILES:
    X, y, indiv = build(pg, p)
    base = y.mean()
    print(f"\n=== percentile {p}: 개인 Strong Dislike 비율 {indiv:.3f}, 그룹 위험 비율(1명 이상) {base:.3f}")
    rule_auc = roc_auc_score(y, X["n_below"] - 1e-3 * X["margin_min"])

    def record(structure, fs, cname, oof):
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        row = dict(percentile=p, structure=structure, features=fs, classifier=cname, logloss=log_loss(y, oof),
                   auc=roc_auc_score(y, oof), pr_auc=average_precision_score(y, oof), brier=brier_score_loss(y, oof),
                   base_rate=base, indiv_dislike_rate=indiv, rule_auc=rule_auc,
                   logloss_constant=log_loss(y, np.full(len(y), base)))
        search.append(row)
        print(f"  {structure:6s} {fs:15s} {cname:24s} logloss {row['logloss']:.4f}  AUC {row['auc']:.4f}  "
              f"PR-AUC {row['pr_auc']:.4f}")

    # (가) 그룹 분류기: 4명을 한 번에 보고 '누군가 Strong Dislike' 확률을 바로 예측
    for (fs, feats), (cname, make) in itertools.product(FEATURE_SETS.items(), CLASSIFIERS.items()):
        oof = np.zeros(len(y))
        for f in range(5):
            a, b = pg_fold != f, pg_fold == f
            clf = make().fit(X.loc[a, feats], y[a])
            oof[b] = clf.predict_proba(X.loc[b, feats])[:, 1]
        record("group", fs, cname, oof)

    # (나) 개인 분류기: 한 명씩 'Strong Dislike 확률' p_i 를 구한 뒤 1-(1-p1)(1-p2)(1-p3)(1-p4) 로 합침
    Xi, yi, Xg = build_indiv(p)
    for cname, make in INDIV_CLASSIFIERS.items():
        oof = np.zeros(len(y))
        for f in range(5):
            clf = make().fit(Xi.loc[pool.fold.to_numpy() != f, INDIV_FEATURES], yi[pool.fold.to_numpy() != f])
            b = pg_fold == f
            pi = np.column_stack([clf.predict_proba(xg.loc[b, INDIV_FEATURES])[:, 1] for xg in Xg])
            oof[b] = 1 - np.prod(1 - pi, axis=1)
        record("indiv", "I_member", cname, oof)

    df_p = pd.DataFrame([s for s in search if s["percentile"] == p])
    per_p[p] = df_p.loc[df_p.logloss.idxmin()].to_dict()

search_df = pd.DataFrame(search)
search_df.to_csv(RESULTS / "stage2_risk_search.csv", index=False)
best_rows = pd.DataFrame(per_p.values())
best_rows.to_csv(RESULTS / "stage2_best_per_percentile.csv", index=False)
print("\n각 percentile 최고 모델:\n", best_rows[["percentile", "structure", "features", "classifier", "logloss", "auc", "pr_auc",
                                            "base_rate", "indiv_dislike_rate", "rule_auc"]].to_string(index=False))

# percentile 선택: AUC 최대, 0.005 이내 동률이면 작은 percentile
top_auc = best_rows.auc.max()
p_star = int(best_rows[best_rows.auc >= top_auc - 0.005].percentile.min())
best = per_p[p_star]
print(f"\n선택된 percentile = {p_star}, 구조 = {best['structure']}, 분류기 = {best['classifier']}, 피처 = {best['features']}")

# 선택된 설정으로 전체 데이터에 다시 학습 -> 고정
if best["structure"] == "group":
    X, y, _ = build(pg, p_star)
    feats = FEATURE_SETS[best["features"]]
    final_clf = CLASSIFIERS[best["classifier"]]().fit(X[feats], y)
else:
    Xi, yi, _ = build_indiv(p_star)
    feats = INDIV_FEATURES
    final_clf = CLASSIFIERS[best["classifier"]]().fit(Xi[feats], yi)
with open(RESULTS / "risk_model.pkl", "wb") as f:
    pickle.dump(dict(model=final_clf, features=feats, kind=best["structure"]), f)
save_json(dict(
    percentile=p_star, structure=best["structure"], classifier=best["classifier"], feature_set=best["features"], features=feats,
    cv_logloss=best["logloss"], cv_auc=best["auc"], cv_pr_auc=best["pr_auc"], base_rate=best["base_rate"],
    indiv_dislike_rate=best["indiv_dislike_rate"], n_pseudo_groups=int(len(pg)),
    rule="percentile 내: CV log loss 최소 / percentile 간: AUC 최대(0.005 이내면 작은 p)",
), "stage2_selection.json")


# ================================================================ lambda 고르기용 추가 연습 그룹 (교차 적합)
# 공식 Validation 그룹은 49개(사용자 36명)뿐이라 lambda 가 '운'에 좌우될 수 있다.
# 그래서 공식 그룹과 같은 방식(4명이 Validation 기간에 공통으로 본 영화 5편 이상)으로 연습 그룹을 더 만든다.
# 단, 이 그룹들의 위험 점수는 '그 사람들을 한 번도 학습에 쓰지 않은' 분류기로 계산해야 공정하다:
#   사용자를 무작위로 반반(H0/H1) 나눔 -> 같은 반 4명으로만 그룹 구성
#   -> H0 그룹은 H1 사람들로만 학습한 분류기로, H1 그룹은 H0 분류기로 위험 점수 계산 (cross-fitting)
#   (최종 고정 분류기는 위에서 전체로 학습한 것. 설정은 동일)
MIN_COMMON, MAX_EXTRA_PER_HALF = 5, 300
half_rng = np.random.default_rng(SEED + 1)
pool_users = np.array(sorted(pool.user_id.unique()))
half_of = dict(zip(pool_users, half_rng.integers(0, 2, len(pool_users))))
th_star = user_thresholds(tr, p_star)
sets = {u: set(d.movie_id) for u, d in pool.groupby("user_id")}


def find_groups(users, rng, limit):
    users = sorted(users, key=lambda u: -len(sets[u]))
    found = set()
    for a in users:
        partners = [b for b in users if b != a and len(sets[a] & sets[b]) >= MIN_COMMON][:40]
        for trio in itertools.combinations(partners, 3):
            common = sets[a] & sets[trio[0]] & sets[trio[1]] & sets[trio[2]]
            if len(common) >= MIN_COMMON:
                found.add(tuple(sorted((a, *trio))))
    found = sorted(found)
    pick = rng.permutation(len(found))[:limit]
    return [found[i] for i in pick], len(found)


def fit_risk(pool_sub, pg_sub):
    if best["structure"] == "group":
        Xs, ys, _ = build(pg_sub, p_star)
        return CLASSIFIERS[best["classifier"]]().fit(Xs[feats], ys)
    Xi_s, yi_s, _ = build_indiv(p_star, pool=pool_sub, pg=pg_sub)
    return CLASSIFIERS[best["classifier"]]().fit(Xi_s[feats], yi_s)


from group import risk_proba  # noqa: E402

lookup = pool.set_index(["user_id", "movie_id"])
extra_rows, gid = [], 1000
for h in (0, 1):
    own = [u for u in pool_users if half_of[u] == h]
    other = pool[pool.user_id.map(half_of) != h]
    model_other = dict(model=fit_risk(other, make_pseudo(other, np.random.default_rng(SEED + 2))),
                       features=feats, kind=best["structure"])
    grps, n_found = find_groups(own, np.random.default_rng(SEED + 3 + h), MAX_EXTRA_PER_HALF)
    print(f"반쪽 H{h}: 사용자 {len(own)}명, 가능한 그룹 {n_found}개 중 {len(grps)}개 사용")
    for users in grps:
        movies = sorted(set.intersection(*[sets[u] for u in users]))
        pred = np.array([[lookup.pred[(u, m)] for u in users] for m in movies])
        true = np.array([[lookup.rating_norm[(u, m)] for u in users] for m in movies])
        thr = th_star.thr.loc[list(users)].to_numpy()
        share = th_star.share.loc[list(users)].to_numpy()
        risk_cf = risk_proba(model_other, pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                             np.array(movies), mstats)
        gid += 1
        for j, m in enumerate(movies):
            extra_rows.append(dict(group_id=gid, half=h, users="|".join(map(str, users)), movie_id=m,
                                   **{f"p{i+1}": pred[j, i] for i in range(4)}, **{f"y{i+1}": true[j, i] for i in range(4)},
                                   **{f"thr{i+1}": thr[i] for i in range(4)}, risk=risk_cf[j]))
extra = pd.DataFrame(extra_rows)
extra.to_csv(RESULTS / "stage2_extra_val_groups.csv", index=False)
print(f"추가 연습 그룹 {extra.group_id.nunique()}개 저장 (후보 영화 평균 {extra.groupby('group_id').size().mean():.1f}편, "
      f"사용자 {len(set('|'.join(extra.users.unique()).split('|')))}명)")
