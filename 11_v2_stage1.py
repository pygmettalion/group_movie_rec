"""[2차 버전] Stage 1 다시 고르기 — Validation 으로만.

새 후보
  A. Content 피처: G(장르) / GY(+개봉 연도) / GYC(+평론가 점수) / GYCD(+감독 효과)
  B. MF: 최근 평점에 더 큰 무게 (반감기 half-life: 없음 / 30 / 60 / 120일)
  C. Hybrid 구조: 1차와 같은 H1/H2/H3 + 단일 모델
선택 규칙은 1차와 같다: Validation(교차검증) RMSE 최소, 0.0003 이내면 더 단순한 것.
1차 버전의 Stage 1 도 후보 안에 그대로 들어 있다(G, 반감기 없음, H2) -> 2차가 못 이기면 1차가 다시 뽑힌다.

실행: RESULTS_DIR=results_v2 python 11_v2_stage1.py
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_movielens, mae, movie_genre_table, rmse, save_json  # noqa: E402
from features_v2 import build_table  # noqa: E402
from models import ALSMF, ContentModel, recency_weights  # noqa: E402

r, _ = load_movielens()
tr = r[r.split == "train"]
va = r[(r.split == "validation") & r.user_id.isin(tr.user_id) & r.movie_id.isin(tr.movie_id)]
gt = movie_genre_table(r)
U, M, Y = tr.user_id.values, tr.movie_id.values, tr.rating_norm.values
vU, vM, vY = va.user_id.values, va.movie_id.values, va.rating_norm.values
print(f"Train {len(tr):,} / Validation 채점 {len(va):,}")

# ------------------------------------------------------------------ A. Content 피처 조합
print("\n[Content v2] 피처 조합 x 하이퍼파라미터")
rows, tables = [], {}
for fs, kappa in [("G", None), ("GY", None), ("GYC", None), ("GYCD", 10.0), ("GYCD", 50.0), ("GYCD", 200.0)]:
    table = gt if fs == "G" else build_table(tr, gt, fs, kappa or 50.0)
    tables[(fs, kappa)] = table
    for k_mean, lam_user, lam_global in itertools.product([10, 20, 50], [30, 100, 300], [0.1, 1.0]):
        cm = ContentModel(k_mean, lam_user, lam_global).fit(U, M, Y, table)
        p = cm.predict(vU, vM)
        rows.append(dict(feature_set=fs, kappa=kappa, k_mean=k_mean, lam_user=lam_user, lam_global=lam_global,
                         val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
    best = min([x for x in rows if x["feature_set"] == fs and x["kappa"] == kappa], key=lambda x: x["val_rmse"])
    print(f"  {fs:5s} kappa={kappa}: best val RMSE {best['val_rmse']:.5f}  ({best})")
# 최적값이 격자 끝(k_mean=50 최대, lam_user=30 최소 등)에 걸리면 그 방향으로 넓혀서 다시 확인 (1차와 같은 원칙)
for _ in range(3):
    c_df = pd.DataFrame(rows).sort_values("val_rmse")
    b = c_df.iloc[0]
    km_all = sorted(set(c_df[(c_df.feature_set == b.feature_set)].k_mean))
    lu_all = sorted(set(c_df[(c_df.feature_set == b.feature_set)].lam_user))
    new_km = [b.k_mean * 2, b.k_mean * 4] if b.k_mean == max(km_all) else []
    new_lu = [b.lam_user / 3, b.lam_user / 10] if b.lam_user == min(lu_all) else []
    new_lu += [b.lam_user * 3] if b.lam_user == max(lu_all) else []
    if not new_km and not new_lu:
        break
    kb = None if pd.isna(b.kappa) else b.kappa
    table = tables[(b.feature_set, kb)]
    for k_mean, lam_user in itertools.product(sorted(set(km_all + new_km)), sorted(set(lu_all + new_lu))):
        if ((c_df.feature_set == b.feature_set) & (c_df.k_mean == k_mean) & (c_df.lam_user == lam_user)
                & (c_df.lam_global == b.lam_global) & ((c_df.kappa == kb) if kb is not None else c_df.kappa.isna())).any():
            continue
        cm = ContentModel(k_mean, lam_user, b.lam_global).fit(U, M, Y, table)
        p = cm.predict(vU, vM)
        rows.append(dict(feature_set=b.feature_set, kappa=kb, k_mean=k_mean, lam_user=lam_user,
                         lam_global=b.lam_global, val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
    print("  격자 확장 후 최고:", pd.DataFrame(rows).sort_values("val_rmse").iloc[0].to_dict())
c_df = pd.DataFrame(rows).sort_values("val_rmse")
c_df.to_csv(RESULTS / "v2_stage1_content_search.csv", index=False)
cb = c_df.iloc[0].to_dict()
c_cfg = dict(feature_set=cb["feature_set"], kappa=None if pd.isna(cb["kappa"]) else float(cb["kappa"]),
             k_mean=float(cb["k_mean"]), lam_user=float(cb["lam_user"]), lam_global=float(cb["lam_global"]))
content = ContentModel(c_cfg["k_mean"], c_cfg["lam_user"], c_cfg["lam_global"]).fit(
    U, M, Y, tables[(c_cfg["feature_set"], c_cfg["kappa"])])
c_val = content.predict(vU, vM)
print("Content 선택:", c_cfg, "val RMSE", rmse(vY, c_val))

# ------------------------------------------------------------------ B. 최근 평점 가중 ALS
print("\n[ALS v2] 반감기 x 규제 (k=64: 1차에서 고른 값)")
rows = []
for hl, reg in itertools.product([None, 30, 60, 120], [0.015, 0.02, 0.03]):
    w = recency_weights(tr.timestamp.values, hl)
    m = ALSMF(k=64, reg=reg, epochs=40, seed=SEED).fit(U, M, Y, val=(vU, vM, vY), patience=2, sample_weight=w)
    p = m.predict(vU, vM)
    rows.append(dict(half_life=hl, k=64, reg=reg, best_iter=m.best_epoch, val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
    print("  ", rows[-1])
a_df = pd.DataFrame(rows).sort_values("val_rmse")
a_df.to_csv(RESULTS / "v2_stage1_als_search.csv", index=False)
ab = a_df.iloc[0].to_dict()
# 반감기 있는 후보가 '없음'보다 0.0003 이상 좋아야 채택 (단순한 쪽 우선)
no_decay = a_df[a_df.half_life.isna()].iloc[0].to_dict()
if not pd.isna(ab["half_life"]) and ab["val_rmse"] > no_decay["val_rmse"] - 0.0003:
    ab = no_decay
mf_cfg = dict(type="als", k=64, reg=float(ab["reg"]), epochs=int(ab["best_iter"]), lr=0.0,
              half_life=None if pd.isna(ab["half_life"]) else float(ab["half_life"]))
mf = ALSMF(k=64, reg=mf_cfg["reg"], epochs=mf_cfg["epochs"], seed=SEED).fit(
    U, M, Y, sample_weight=recency_weights(tr.timestamp.values, mf_cfg["half_life"]))
mf_val = mf.predict(vU, vM)
print("MF 선택:", mf_cfg, "val RMSE", rmse(vY, mf_val))

# ------------------------------------------------------------------ C. Hybrid 구조 (1차와 같은 방식)
n_hist = tr.groupby("user_id").size()
EDGES = [0, 50, 200]
bucket = np.digitize(va.user_id.map(n_hist).to_numpy(), EDGES[1:])
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def fit_global(i):
    return min(ALPHAS, key=lambda a: np.mean((vY[i] - (a * mf_val[i] + (1 - a) * c_val[i])) ** 2))


def fit_bucket(i):
    return {b: (fit_global(i[bucket[i] == b]) if (bucket[i] == b).any() else fit_global(i)) for b in range(3)}


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
cv_rows = []
for kind, fitter in FITTERS.items():
    oof = np.zeros(len(vY))
    for a_idx, b_idx in GroupKFold(n_splits=5).split(vY, groups=vU):
        oof[b_idx] = apply(kind, fitter(a_idx), b_idx)
    cv_rows.append(dict(structure=kind, cv_rmse=rmse(vY, oof), cv_mae=mae(vY, oof)))
cv_rows += [dict(structure="MF only", cv_rmse=rmse(vY, mf_val), cv_mae=mae(vY, mf_val)),
            dict(structure="Content only", cv_rmse=rmse(vY, c_val), cv_mae=mae(vY, c_val))]
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(RESULTS / "v2_stage1_structure_compare.csv", index=False)
print("\n", cv_df.to_string(index=False))
order = ["MF only", "Content only", "H1_global", "H3_stack", "H2_bucket"]
scores = cv_df.set_index("structure").cv_rmse
chosen = next(o for o in order if scores[o] <= scores.min() + 0.0003)
full = {k: f(np.arange(len(vY))) for k, f in FITTERS.items()}
sel = dict(version="v2", chosen_model="Hybrid" if chosen.startswith("H") else chosen.split()[0],
           structure=chosen, mf=mf_cfg, content=c_cfg, alpha={"MF only": 1.0, "Content only": 0.0}.get(chosen),
           hybrid_params=None, bucket_edges=EDGES, validation_scores=scores.to_dict(),
           v1_reference_val_rmse=0.22259339380149828,
           rule="교차검증 RMSE 최소, 0.0003 이내면 더 단순한 구조")
if chosen == "H1_global":
    sel["alpha"] = float(full["H1_global"])
elif chosen == "H2_bucket":
    sel["hybrid_params"] = {str(k): float(v) for k, v in full["H2_bucket"].items()}
elif chosen == "H3_stack":
    sel["hybrid_params"] = [float(x) for x in full["H3_stack"]]
save_json(sel, "stage1_selection_final.json")
print("\n2차 Stage 1 선택:", sel)
print(f"1차 Stage 1 Validation RMSE 0.22259 -> 2차 {scores[chosen]:.5f}")
