"""Compare additional Stage 1 candidates -> decide the final Stage 1 structure.

01_stage1_select.py compared the candidates written in the design doc (SGD-MF, Content, Hybrid with a single alpha).
Here we check with data whether 'a more reasonable structure exists'.

 A. MF training method: ALS (alternating exact solves) vs SGD. Plus a k=0 baseline (user/movie biases only, no taste factors).
    -> checks whether latent factors actually help.
 B. Hybrid mixing structure
    H1 global : the same alpha for everyone                          (design-doc default)
    H2 bucket : alpha depends on the number of past ratings (few/medium/many)  ("trust MF more for users with long history")
    H3 stack  : a + b*MF + c*Content  (linear combination that also recalibrates)
    -> Because the mixing ratio is 'learned' on Validation, Validation users are split into 5 folds for fairness:
       the ratio is fit on 4 folds and scored on the remaining fold (cross-validation).
 Selection rule: lowest cross-validated RMSE. If the gap is below 0.0003 (about 0.001 stars), take the simpler structure.
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_json, load_movielens, mae, movie_genre_table, rmse, save_json  # noqa: E402
from models import ALSMF, MF, ContentModel  # noqa: E402

r, _ = load_movielens()
tr = r[r.split == "train"]
va = r[(r.split == "validation") & r.user_id.isin(tr.user_id) & r.movie_id.isin(tr.movie_id)]
gt = movie_genre_table(r)
U, M, Y = tr.user_id.values, tr.movie_id.values, tr.rating_norm.values
vU, vM, vY = va.user_id.values, va.movie_id.values, va.rating_norm.values
base_sel = load_json("stage1_selection.json")

# ------------------------------------------------------------------ A. ALS MF (+ k=0 baseline)
print("[ALS-MF] grid search")
rows = []
for k, reg in itertools.product([0, 4, 8, 16, 32, 64], [0.01, 0.015, 0.02, 0.03, 0.05, 0.1]):
    m = ALSMF(k=k, reg=reg, epochs=60, seed=SEED).fit(U, M, Y, val=(vU, vM, vY), patience=2)
    p = m.predict(vU, vM)
    rows.append(dict(k=k, reg=reg, best_iter=m.best_epoch, val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
    print("  ", rows[-1])
als_df = pd.DataFrame(rows).sort_values("val_rmse")
# If the best k is at the grid edge (64), also try 2x (128). Adopt the larger k only if it improves by >= 0.0003 (prefer simpler, same rule as other choices)
top = als_df.iloc[0]
if top.k == max(r_["k"] for r_ in rows):
    for reg in [top.reg, round(top.reg * 1.25, 4)]:
        m = ALSMF(k=int(top.k * 2), reg=reg, epochs=40, seed=SEED).fit(U, M, Y, val=(vU, vM, vY), patience=2)
        p = m.predict(vU, vM)
        gain = top.val_rmse - rmse(vY, p)
        rows.append(dict(k=int(top.k * 2), reg=reg, best_iter=m.best_epoch, val_rmse=rmse(vY, p), val_mae=mae(vY, p),
                         note=f"격자 끝 확인: k={int(top.k)} 대비 {gain:+.5f}" + (" -> 채택" if gain > 0.0003 else " -> 0.0003 미만, 작은 k 유지")))
        print("  ", rows[-1])
    big = pd.DataFrame(rows)
    big = big[big.k == top.k * 2]
    als_df = pd.DataFrame(rows).sort_values("val_rmse")
    if not (top.val_rmse - big.val_rmse.min() > 0.0003):
        als_df = pd.concat([als_df[als_df.k != top.k * 2], als_df[als_df.k == top.k * 2]])   # put the smaller k on top
als_df.to_csv(RESULTS / "stage1b_als_search.csv", index=False)
als_best = als_df.iloc[0].to_dict()
bias_best = als_df[als_df.k == 0].iloc[0].to_dict()
print("ALS 최고:", als_best, "\nk=0(편향만) 최고:", bias_best)

sgd = base_sel["mf"]
sgd_rmse = base_sel["validation_rmse"]["MF"]
if als_best["val_rmse"] < sgd_rmse:
    mf_choice = dict(type="als", k=int(als_best["k"]), reg=als_best["reg"], epochs=int(als_best["best_iter"]), lr=0.0)
    mf_model = ALSMF(k=mf_choice["k"], reg=mf_choice["reg"], epochs=mf_choice["epochs"], seed=SEED).fit(U, M, Y)
else:
    mf_choice = dict(type="sgd", **sgd)
    mf_model = MF(k=sgd["k"], lr=sgd["lr"], reg=sgd["reg"], epochs=sgd["epochs"], seed=SEED).fit(U, M, Y)
mf_val = mf_model.predict(vU, vM)
print("MF 선택:", mf_choice, "val RMSE", rmse(vY, mf_val))

# Content: the best value from 01 hit the 'edge' of the search range (k_mean=20, lam_user=100), so widen the range and re-check
print("\n[Content] 범위 확장 탐색 (01 최적값이 격자 끝이었음)")
c = base_sel["content"]
c_rows = [dict(**c, val_rmse=base_sel["validation_rmse"]["Content"])]
for k_mean, lam_user in itertools.product([20, 50, 100, 200], [100, 300, 1000, 3000]):
    cm = ContentModel(k_mean, lam_user, c["lam_global"]).fit(U, M, Y, gt)
    c_rows.append(dict(k_mean=k_mean, lam_user=lam_user, lam_global=c["lam_global"],
                       val_rmse=rmse(vY, cm.predict(vU, vM))))
c_ext = pd.DataFrame(c_rows).drop_duplicates(["k_mean", "lam_user", "lam_global"]).sort_values("val_rmse")
c_ext.to_csv(RESULTS / "stage1b_content_extended.csv", index=False)
print(c_ext.head(6).to_string(index=False))
c = {k: float(v) for k, v in c_ext.iloc[0][["k_mean", "lam_user", "lam_global"]].items()}
content = ContentModel(c["k_mean"], c["lam_user"], c["lam_global"]).fit(U, M, Y, gt)
c_val = content.predict(vU, vM)

# ------------------------------------------------------------------ B. Hybrid structure comparison (cross-validation)
n_hist = tr.groupby("user_id").size()
BUCKETS = [0, 50, 200, np.inf]           # number of past ratings: <50 / 50~199 / 200+
bucket = np.digitize(va.user_id.map(n_hist).to_numpy(), BUCKETS[1:-1])
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def fit_global(i):
    return min(ALPHAS, key=lambda a: np.mean((vY[i] - (a * mf_val[i] + (1 - a) * c_val[i])) ** 2))


def fit_bucket(i):
    out = {}
    for b in range(len(BUCKETS) - 1):
        j = i[bucket[i] == b]
        out[b] = fit_global(j) if len(j) else fit_global(i)
    return out


def fit_stack(i):
    X = np.column_stack([np.ones(len(i)), mf_val[i], c_val[i]])
    return np.linalg.lstsq(X, vY[i], rcond=None)[0]


def apply(kind, prm, i):
    if kind == "H1_global":
        return prm * mf_val[i] + (1 - prm) * c_val[i]
    if kind == "H2_bucket":
        a = np.array([prm[b] for b in bucket[i]])
        return a * mf_val[i] + (1 - a) * c_val[i]
    return prm[0] + prm[1] * mf_val[i] + prm[2] * c_val[i]


FITTERS = {"H1_global": fit_global, "H2_bucket": fit_bucket, "H3_stack": fit_stack}
cv = GroupKFold(n_splits=5)
cv_rows = []
for kind, fitter in FITTERS.items():
    oof = np.zeros(len(vY))
    for a_idx, b_idx in cv.split(vY, groups=vU):
        oof[b_idx] = apply(kind, fitter(a_idx), b_idx)
    full = fitter(np.arange(len(vY)))
    cv_rows.append(dict(structure=kind, cv_rmse=rmse(vY, oof), cv_mae=mae(vY, oof),
                        params=str(full if not isinstance(full, np.ndarray) else np.round(full, 4).tolist())))
# single models go in the same table too (nothing is 'learned' on Validation, so they are scored as is)
cv_rows += [dict(structure="MF only", cv_rmse=rmse(vY, mf_val), cv_mae=mae(vY, mf_val), params=str(mf_choice)),
            dict(structure="Content only", cv_rmse=rmse(vY, c_val), cv_mae=mae(vY, c_val), params=str(c)),
            dict(structure="k=0 편향만(참고)", cv_rmse=bias_best["val_rmse"], cv_mae=bias_best["val_mae"], params="")]
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(RESULTS / "stage1b_structure_compare.csv", index=False)
print("\n", cv_df.to_string(index=False))

# selection: simplicity order = MF only / Content only (0 params) < H1 (1) < H3 (3) < H2 (number of buckets)
order = ["MF only", "Content only", "H1_global", "H3_stack", "H2_bucket"]
scores = cv_df.set_index("structure").cv_rmse
best_val = min(scores[o] for o in order)
chosen = next(o for o in order if scores[o] <= best_val + 0.0003)
print("\n최종 Stage 1 구조:", chosen)

full_params = {k: FITTERS[k](np.arange(len(vY))) for k in FITTERS}
sel = dict(
    chosen_model={"MF only": "MF", "Content only": "Content", "H1_global": "Hybrid", "H2_bucket": "Hybrid",
                  "H3_stack": "Hybrid"}[chosen],
    structure=chosen, mf=mf_choice, content=c,
    alpha={"MF only": 1.0, "Content only": 0.0}.get(chosen, None),
    hybrid_params=None,
    bucket_edges=[0, 50, 200],
    validation_scores=cv_df.set_index("structure").cv_rmse.to_dict(),
    rule="교차검증 RMSE 최소, 0.0003 이내면 더 단순한 구조",
)
if chosen == "H1_global":
    sel["alpha"] = float(full_params["H1_global"])
elif chosen == "H2_bucket":
    sel["hybrid_params"] = {str(k): float(v) for k, v in full_params["H2_bucket"].items()}
elif chosen == "H3_stack":
    sel["hybrid_params"] = [float(x) for x in full_params["H3_stack"]]
save_json(sel, "stage1_selection_final.json")
print(sel)
