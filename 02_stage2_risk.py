"""Stage 2: choose the Strong Dislike threshold (percentile) + the Group Risk classifier.

In plain words
  Build a model that predicts "if these 4 people watch this movie, the probability that someone says 'that was awful!'".

How is the training data built?
  - There are only 49 official evaluation groups, far too few to train a classifier.
  - So in the Validation period, 4 people who watched the same movie are grouped at random into practice groups (pseudo-groups).
  - The 36 users of the official evaluation groups are fully excluded here -> no overlap with the data later used to choose lambda.
  - The individual preference score is the output of the Stage 1 model trained on Train only (= a prediction made without knowing that rating).

What is compared?
  - Strong Dislike percentile candidates: 5, 10, 15, 20, 25
  - Classifiers: Logistic Regression / HistGradientBoosting / RandomForest (a few settings each)
  - Feature sets: F1 (prediction only) / F2 (+ distance to the personal Strong Dislike threshold) / F3 (+ the movie's past rating statistics)
  - Scoring: 5-fold cross-validation split by movie (the same movie never appears in both training and scoring)

Selection rules (fixed before looking at results)
  1) Within each percentile: the (classifier, setting, feature set) with the lowest cross-validated log loss
     -> the probability itself enters the final score, so log loss (how well-calibrated the probability is) is used.
  2) Across percentiles: the value with the highest ROC-AUC of the model above (how well risky movies are ranked, independent of the base rate).
     If the AUC gap is within 0.005, pick the smaller percentile (a 'stronger' dislike).
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

MIN_HISTORY = 10          # same condition as group users (at least 10 past ratings)
GROUPS_PER_MOVIE = 20     # maximum number of pseudo-groups per movie (so popular movies do not dominate the data)

r, groups = load_movielens()
tr = r[r.split == "train"]
va = r[r.split == "validation"]
gt = movie_genre_table(r)
sel = load_json("stage1_selection_final.json")

# ---------------------------------------------------------------- Stage 1 predictions (trained on Train only)
print("Stage 1 모델을 Train 으로 학습 ->", sel["structure"])
s1 = Stage1Predictor(sel, gt, seed=SEED).fit(tr)
counts = tr.groupby("user_id").size()
eval_users = set(groups[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())

va_w = va[va.user_id.isin(counts[counts >= MIN_HISTORY].index) & va.movie_id.isin(tr.movie_id)].copy()
va_w["pred"] = s1.predict(va_w.user_id.values, va_w.movie_id.values)
va_w[["user_id", "movie_id", "rating_norm", "pred"]].to_csv(RESULTS / "val_stage1_predictions.csv", index=False)

# ---------------------------------------------------------------- build pseudo-groups
pool = va_w[~va_w.user_id.isin(eval_users)]
rng = np.random.default_rng(SEED)
cols = ["movie_id", "u1", "u2", "u3", "u4", "y1", "y2", "y3", "y4", "p1", "p2", "p3", "p4"]


def make_pseudo(pool_df, rng):
    """Randomly group 4 people who watched the same movie in the Validation period (at most GROUPS_PER_MOVIE per movie)."""
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

# 5-fold by movie: the same movie is always in the same fold (the group and individual classifiers use exactly the same split)
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
    # individual features for each of the 4 pseudo-group members (to combine risk as a product over the 4 people)
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

    # (a) group classifier: looks at all 4 people at once and directly predicts the probability that 'someone strongly dislikes it'
    for (fs, feats), (cname, make) in itertools.product(FEATURE_SETS.items(), CLASSIFIERS.items()):
        oof = np.zeros(len(y))
        for f in range(5):
            a, b = pg_fold != f, pg_fold == f
            clf = make().fit(X.loc[a, feats], y[a])
            oof[b] = clf.predict_proba(X.loc[b, feats])[:, 1]
        record("group", fs, cname, oof)

    # (b) individual classifier: get each person's 'Strong Dislike probability' p_i, then combine as 1-(1-p1)(1-p2)(1-p3)(1-p4)
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

# percentile choice: maximum AUC; ties within 0.005 go to the smaller percentile
top_auc = best_rows.auc.max()
p_star = int(best_rows[best_rows.auc >= top_auc - 0.005].percentile.min())
best = per_p[p_star]
print(f"\n선택된 percentile = {p_star}, 구조 = {best['structure']}, 분류기 = {best['classifier']}, 피처 = {best['features']}")

# refit the chosen setting on all data -> freeze
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


# ================================================================ extra practice groups for choosing lambda (cross-fitted)
# The official Validation groups are only 49 (36 users), so lambda could depend on 'luck'.
# So more practice groups are built the same way as the official ones (4 people with >= 5 common movies in the Validation period).
# However, for fairness their risk scores must come from a classifier that 'never saw those people in training':
#   split users randomly in half (H0/H1) -> build groups only from 4 people in the same half
#   -> H0 groups are scored by a classifier trained only on H1 people, and H1 groups by the H0 classifier (cross-fitting)
#   (the final frozen classifier is the one trained on all data above, with the same setting)
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
