"""[3차] Stage 1 다시 고르기 — Validation 으로만 (Train 학습 -> Validation 채점).

후보
  - MF: 1번 학습 vs 여러 번(5번, 시작점만 다르게) 학습해서 평균 (앙상블)
  - Item-kNN: 이웃 수 k, 유사도 당김(sim_shrink) 격자
  - 섞기 모델: H2(2차 방식) / LIN3 / LIN3_B / GBM  -> 사용자 단위 5-fold 교차검증 RMSE
선택 규칙: 교차검증 RMSE 최소, 0.0003 이내면 더 단순한 것 (1·2차와 같음).

실행: RESULTS_DIR=results_v3 python 31_v3_stage1.py
"""
import itertools
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
from common import ROOT, RESULTS, SEED, load_movielens, mae, movie_genre_table, rmse, save_json  # noqa: E402
from features_v2 import build_table  # noqa: E402
from knn import ItemKNN  # noqa: E402
from models import ALSMF, ContentModel  # noqa: E402
from stage1_v3 import BLEND_FEATURES, apply_blender, blend_features  # noqa: E402

r, _ = load_movielens()
tr = r[r.split == "train"]
va = r[(r.split == "validation") & r.user_id.isin(tr.user_id) & r.movie_id.isin(tr.movie_id)].reset_index(drop=True)
gt = movie_genre_table(r)
U, M, Y = tr.user_id.values, tr.movie_id.values, tr.rating_norm.values
vU, vM, vY = va.user_id.values, va.movie_id.values, va.rating_norm.values
v2 = json.load(open(ROOT / "results_v2" / "stage1_selection_final.json", encoding="utf-8"))
mf_cfg, c_cfg = dict(v2["mf"]), dict(v2["content"])

# ------------------------------------------------------------------ MF: 1번 vs 5번 평균
print("[MF] 앙상블")
mf_preds = []
for s in range(5):
    m = ALSMF(k=mf_cfg["k"], reg=mf_cfg["reg"], epochs=mf_cfg["epochs"], seed=SEED + s).fit(U, M, Y)
    mf_preds.append(m.predict(vU, vM))
    print(f"  seed {s}: {rmse(vY, mf_preds[-1]):.5f} / 평균 {s + 1}개: {rmse(vY, np.mean(mf_preds, 0)):.5f}")
mf_one, mf_bag = mf_preds[0], np.mean(mf_preds, 0)
mf_rows = [dict(n_bag=1, val_rmse=rmse(vY, mf_one)), dict(n_bag=5, val_rmse=rmse(vY, mf_bag))]
pd.DataFrame(mf_rows).to_csv(RESULTS / "v3_stage1_mf_bag.csv", index=False)

# ------------------------------------------------------------------ Content (2차에서 고른 설정 그대로)
table = build_table(tr, gt, c_cfg["feature_set"], c_cfg["kappa"] or 50.0)
c_val = ContentModel(c_cfg["k_mean"], c_cfg["lam_user"], c_cfg["lam_global"]).fit(U, M, Y, table).predict(vU, vM)

# ------------------------------------------------------------------ Item-kNN 격자
print("[Item-kNN]")
knn_rows, knn_preds = [], {}
for k, ss in itertools.product([20, 40, 80, 160], [25.0, 100.0]):
    p = ItemKNN(k=k, sim_shrink=ss).fit(U, M, Y).predict(vU, vM)
    knn_preds[(k, ss)] = p
    knn_rows.append(dict(k=k, sim_shrink=ss, val_rmse=rmse(vY, p)))
    print("  ", knn_rows[-1])
knn_df = pd.DataFrame(knn_rows).sort_values("val_rmse")
knn_df.to_csv(RESULTS / "v3_stage1_knn_search.csv", index=False)
kb = knn_df.iloc[0]
knn_cfg = dict(k=int(kb.k), sim_shrink=float(kb.sim_shrink))
knn_val = knn_preds[(knn_cfg["k"], knn_cfg["sim_shrink"])]
print("kNN 선택:", knn_cfg, kb.val_rmse)

# ------------------------------------------------------------------ 섞기 모델 비교 (사용자 단위 교차검증)
n_user = tr.groupby("user_id").size().to_dict()
n_item = tr.groupby("movie_id").size().to_dict()
u_std = tr.groupby("user_id").rating_norm.std().to_dict()
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def X_for(mfp):
    return blend_features(mfp, c_val, knn_val, vU, vM, n_user, n_item, u_std)


def fit_h2(X, y):
    b = np.digitize(np.expm1(X.log_nu.to_numpy()), [50, 200])
    out = {}
    for j in range(3):
        k = b == j
        if not k.any():
            k = np.ones(len(y), bool)
        out[str(j)] = float(min(ALPHAS, key=lambda a: np.mean((y[k] - (a * X.mf.to_numpy()[k] + (1 - a) * X.content.to_numpy()[k])) ** 2)))
    return out


def fit_lin3(X, y):
    A = np.column_stack([np.ones(len(y)), X.mf, X.content, X.knn])
    return [float(v) for v in np.linalg.lstsq(A, y, rcond=None)[0]]


def fit_lin3b(X, y):
    b = np.digitize(np.expm1(X.log_nu.to_numpy()), [50, 200])
    return {str(j): fit_lin3(X[b == j] if (b == j).any() else X, y[b == j] if (b == j).any() else y) for j in range(3)}


def make_gbm(lr, leaves):
    def fit(X, y):
        return HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300, l2_regularization=1.0,
                                             min_samples_leaf=100, random_state=SEED).fit(X[BLEND_FEATURES], y)
    return fit


CANDIDATES = [  # (이름, MF 종류, 섞기 종류, 학습 함수)  -- 단순한 순서
    ("H2 + MF 1번", "one", "H2", fit_h2),
    ("H2 + MF 5번 평균", "bag", "H2", fit_h2),
    ("LIN3 + MF 5번 평균", "bag", "LIN3", fit_lin3),
    ("LIN3_B + MF 5번 평균", "bag", "LIN3_B", fit_lin3b),
] + [(f"GBM(lr={lr},leaves={lv}) + MF 5번 평균", "bag", "GBM", make_gbm(lr, lv)) for lr, lv in [(0.05, 15), (0.05, 31), (0.1, 15)]]

rows = []
for name, mfk, kind, fit in CANDIDATES:
    X = X_for(mf_one if mfk == "one" else mf_bag)
    oof = np.zeros(len(vY))
    for a_idx, b_idx in GroupKFold(n_splits=5).split(X, groups=vU):
        oof[b_idx] = apply_blender(kind, fit(X.iloc[a_idx], vY[a_idx]), X.iloc[b_idx])
    rows.append(dict(candidate=name, mf=mfk, blender=kind, cv_rmse=rmse(vY, oof), cv_mae=mae(vY, oof)))
    print(f"  {name:32s} CV RMSE {rows[-1]['cv_rmse']:.5f}")
cmp = pd.DataFrame(rows)
cmp.to_csv(RESULTS / "v3_stage1_blender_compare.csv", index=False)
best = cmp.cv_rmse.min()
pick = next(i for i in range(len(rows)) if rows[i]["cv_rmse"] <= best + 0.0003)
name, mfk, kind, fit = CANDIDATES[pick]
X = X_for(mf_one if mfk == "one" else mf_bag)
params = fit(X, vY)
sel = dict(version="v3", blender=kind, candidate=name,
           mf=dict(type="als", k=mf_cfg["k"], reg=mf_cfg["reg"], epochs=mf_cfg["epochs"], n_bag=1 if mfk == "one" else 5),
           content=c_cfg, knn=knn_cfg, blender_params=None if kind == "GBM" else params,
           validation_cv_rmse=float(rows[pick]["cv_rmse"]), v2_reference=0.2215529315606526,
           rule="교차검증 RMSE 최소, 0.0003 이내면 더 단순한 것")
if kind == "GBM":
    with open(RESULTS / "stage1_v3_blender.pkl", "wb") as f:
        pickle.dump(params, f)
save_json(sel, "stage1_selection_final.json")
print("\n3차 Stage 1 선택:", name, f"CV RMSE {rows[pick]['cv_rmse']:.5f} (2차 0.22155)")
