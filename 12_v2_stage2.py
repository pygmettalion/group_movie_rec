"""[2차 버전] Strong Dislike 기준 + 위험 모델 다시 고르기 — Validation 으로만.

1차와 같은 절차(가짜 그룹, 영화 단위 5-fold, 같은 선택 규칙)에
새 피처 조합(I_meta, I_gd, I_meta_gd)을 후보로 추가했다. 1차 피처(I_member)도 후보에 그대로 있다.
1차에서 '개인 확률 -> 4명 합치기' 구조가 모든 percentile 에서 그룹 분류기를 이겨서, 2차는 그 구조만 비교한다.

실행: RESULTS_DIR=results_v2 python 12_v2_stage2.py
"""
import itertools
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_json, load_movielens, movie_genre_table, save_json  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import PERCENTILES, movie_stats, user_thresholds  # noqa: E402
from risk_v2 import INDIV_V2_SETS, RiskContext, indiv_features_v2, risk_proba_v2  # noqa: E402

MIN_HISTORY, GROUPS_PER_MOVIE, ALPHAS = 10, 20, [3.0, 10.0]
r, groups = load_movielens()
tr, va = r[r.split == "train"], r[r.split == "validation"]
gt = movie_genre_table(r)
genre_lookup = r.drop_duplicates("movie_id").set_index("movie_id").genres.to_dict()
sel = load_json("stage1_selection_final.json")
KAPPA = sel["content"].get("kappa") or 50.0

print("Stage 1 (2차) 을 Train 으로 학습 ->", sel["structure"], sel["content"]["feature_set"], "half_life", sel["mf"].get("half_life"))
s1 = Stage1Predictor(sel, gt, seed=SEED).fit(tr)
counts = tr.groupby("user_id").size()
eval_users = set(groups[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())
va_w = va[va.user_id.isin(counts[counts >= MIN_HISTORY].index) & va.movie_id.isin(tr.movie_id)].copy()
va_w["pred"] = s1.predict(va_w.user_id.values, va_w.movie_id.values)
va_w[["user_id", "movie_id", "rating_norm", "pred"]].to_csv(RESULTS / "val_stage1_predictions.csv", index=False)

pool = va_w[~va_w.user_id.isin(eval_users)].reset_index(drop=True)
rng = np.random.default_rng(SEED)
cols = ["movie_id", "u1", "u2", "u3", "u4", "y1", "y2", "y3", "y4", "p1", "p2", "p3", "p4"]


def make_pseudo(pool_df, rng):
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
UC, YC, PC = ["u1", "u2", "u3", "u4"], ["y1", "y2", "y3", "y4"], ["p1", "p2", "p3", "p4"]
mstats = movie_stats(tr)
movie_fold = {m: i % 5 for i, m in enumerate(rng.permutation(pool.movie_id.unique()))}
pg_fold = pg.movie_id.map(movie_fold).to_numpy()
pool_fold = pool.movie_id.map(movie_fold).to_numpy()
print(f"가짜 그룹 {len(pg):,}개, 개인 학습 행 {len(pool):,}개")

CLASSIFIERS = {
    "LogReg(C=1)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
    "HGB(lr=0.03,leaves=31)": lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=31, max_iter=200,
                                                                     l2_regularization=1.0, random_state=SEED),
    "HGB(lr=0.1,leaves=7)": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=7, max_iter=200,
                                                                   l2_regularization=1.0, random_state=SEED),
}


def build(p, alpha, pool_df=pool, pg_df=pg):
    th = user_thresholds(tr, p)
    ctx = RiskContext(tr, th.thr, genre_lookup, alpha, tr, KAPPA)
    Xi = indiv_features_v2(pool_df.pred.to_numpy(), pool_df.user_id.map(th.thr).to_numpy(),
                           pool_df.user_id.map(th.share).to_numpy(), pool_df.movie_id.to_numpy(),
                           pool_df.user_id.to_numpy(), mstats, ctx)
    yi = (pool_df.rating_norm.to_numpy() <= pool_df.user_id.map(th.thr).to_numpy()).astype(int)
    Xg = [indiv_features_v2(pg_df[pc].to_numpy(), pg_df[uc].map(th.thr).to_numpy(), pg_df[uc].map(th.share).to_numpy(),
                            pg_df.movie_id.to_numpy(), pg_df[uc].to_numpy(), mstats, ctx) for pc, uc in zip(PC, UC)]
    thr4 = np.column_stack([pg_df[c].map(th.thr).to_numpy() for c in UC])
    yg = (pg_df[YC].to_numpy() <= thr4).any(1).astype(int)
    return Xi, yi, Xg, yg, float((pg_df[YC].to_numpy() <= thr4).mean()), ctx, th


search, per_p = [], {}
for p in PERCENTILES:
    for alpha in ALPHAS:
        Xi, yi, Xg, y, indiv_rate, _, _ = build(p, alpha)
        base = y.mean()
        for fs, feats in INDIV_V2_SETS.items():
            if alpha != ALPHAS[0] and "u_gdis" not in feats:
                continue   # alpha 는 장르 Strong Dislike 피처에만 영향
            for cname, make in CLASSIFIERS.items():
                oof = np.zeros(len(y))
                for f in range(5):
                    clf = make().fit(Xi.loc[pool_fold != f, feats], yi[pool_fold != f])
                    b = pg_fold == f
                    pi = np.column_stack([clf.predict_proba(xg.loc[b, feats])[:, 1] for xg in Xg])
                    oof[b] = 1 - np.prod(1 - pi, axis=1)
                oof = np.clip(oof, 1e-6, 1 - 1e-6)
                row = dict(percentile=p, features=fs, alpha=alpha if "u_gdis" in feats else None, classifier=cname,
                           logloss=log_loss(y, oof), auc=roc_auc_score(y, oof), pr_auc=average_precision_score(y, oof),
                           brier=brier_score_loss(y, oof), base_rate=base, indiv_dislike_rate=indiv_rate)
                search.append(row)
                print(f"  p={p:2d} {fs:10s} a={row['alpha']} {cname:24s} logloss {row['logloss']:.4f} AUC {row['auc']:.4f}")
    df_p = pd.DataFrame([s for s in search if s["percentile"] == p])
    per_p[p] = df_p.loc[df_p.logloss.idxmin()].to_dict()

pd.DataFrame(search).to_csv(RESULTS / "v2_stage2_risk_search.csv", index=False)
best_rows = pd.DataFrame(per_p.values())
best_rows.to_csv(RESULTS / "v2_stage2_best_per_percentile.csv", index=False)
print("\n", best_rows[["percentile", "features", "alpha", "classifier", "logloss", "auc", "base_rate"]].to_string(index=False))
top = best_rows.auc.max()
p_star = int(best_rows[best_rows.auc >= top - 0.005].percentile.min())
best = per_p[p_star]
alpha_star = best["alpha"] if best["alpha"] is not None and not pd.isna(best["alpha"]) else ALPHAS[0]
feats = INDIV_V2_SETS[best["features"]]
print(f"\n선택: percentile {p_star}, 피처 {best['features']}, alpha {best['alpha']}, {best['classifier']}")

Xi, yi, _, _, _, ctx_star, th_star = build(p_star, alpha_star)
final_clf = CLASSIFIERS[best["classifier"]]().fit(Xi[feats], yi)
RISK = dict(model=final_clf, features=feats, kind="indiv_v2", alpha=alpha_star, kappa=KAPPA, percentile=p_star)
with open(RESULTS / "risk_model.pkl", "wb") as f:
    pickle.dump(RISK, f)
save_json(dict(version="v2", percentile=p_star, structure="indiv_v2", feature_set=best["features"], features=feats,
               alpha=alpha_star, kappa=KAPPA, classifier=best["classifier"], cv_logloss=best["logloss"],
               cv_auc=best["auc"], base_rate=best["base_rate"], indiv_dislike_rate=best["indiv_dislike_rate"],
               n_pseudo_groups=int(len(pg)),
               rule="percentile 내: CV log loss 최소 / percentile 간: AUC 최대(0.005 이내면 작은 p)"),
          "stage2_selection.json")

# ================================================================ lambda 용 추가 연습 그룹 (1차와 같은 교차 적합)
MIN_COMMON, MAX_EXTRA = 5, 300
pool_users = np.array(sorted(pool.user_id.unique()))
half_of = dict(zip(pool_users, np.random.default_rng(SEED + 1).integers(0, 2, len(pool_users))))
sets = {u: set(d.movie_id) for u, d in pool.groupby("user_id")}
lookup = pool.set_index(["user_id", "movie_id"])


def find_groups(users, rng, limit):
    users = sorted(users, key=lambda u: -len(sets[u]))
    found = set()
    for a in users:
        partners = [b for b in users if b != a and len(sets[a] & sets[b]) >= MIN_COMMON][:40]
        for trio in itertools.combinations(partners, 3):
            if len(sets[a] & sets[trio[0]] & sets[trio[1]] & sets[trio[2]]) >= MIN_COMMON:
                found.add(tuple(sorted((a, *trio))))
    found = sorted(found)
    return [found[i] for i in rng.permutation(len(found))[:limit]]


extra_rows, gid = [], 1000
for h in (0, 1):
    own = [u for u in pool_users if half_of[u] == h]
    other = pool[pool.user_id.map(half_of) != h]
    Xo, yo, _, _, _, _, _ = build(p_star, alpha_star, pool_df=other, pg_df=pg.iloc[:0])
    model_other = dict(model=CLASSIFIERS[best["classifier"]]().fit(Xo[feats], yo), features=feats)
    for users in find_groups(own, np.random.default_rng(SEED + 3 + h), MAX_EXTRA):
        movies = sorted(set.intersection(*[sets[u] for u in users]))
        pred = np.array([[lookup.pred[(u, m)] for u in users] for m in movies])
        true = np.array([[lookup.rating_norm[(u, m)] for u in users] for m in movies])
        thr = th_star.thr.loc[list(users)].to_numpy()
        share = th_star.share.loc[list(users)].to_numpy()
        risk_cf = risk_proba_v2(model_other, pred, np.tile(thr, (len(movies), 1)), np.tile(share, (len(movies), 1)),
                                np.array(movies), list(users), mstats, ctx_star)
        gid += 1
        for j, m in enumerate(movies):
            extra_rows.append(dict(group_id=gid, half=h, users="|".join(map(str, users)), movie_id=m,
                                   **{f"p{i+1}": pred[j, i] for i in range(4)}, **{f"y{i+1}": true[j, i] for i in range(4)},
                                   **{f"thr{i+1}": thr[i] for i in range(4)}, risk=risk_cf[j]))
extra = pd.DataFrame(extra_rows)
extra.to_csv(RESULTS / "stage2_extra_val_groups.csv", index=False)
print(f"추가 연습 그룹 {extra.group_id.nunique()}개")
