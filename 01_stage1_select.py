"""Stage 1: 개인 선호 예측 모델 고르기 (Train 으로 공부, Validation 으로 채점).

순서
 1. MF 하이퍼파라미터: 넓게(coarse) 훑고 -> 제일 좋은 곳 주변을 좁게(fine) 다시 비교
 2. Content 하이퍼파라미터 격자 탐색
 3. Hybrid 섞는 비율 alpha 를 0~1 사이 0.05 간격으로 비교
 4. MF / Content / Hybrid 중 Validation RMSE 가 가장 낮은 것을 최종 Stage 1 모델로 선택

공정한 비교를 위해 세 모델 모두 '같은 Validation 평점 묶음'으로 채점한다:
 = Validation 기간 평점 중, Train 에 기록이 있는 사용자 & Train 에 등장한 영화
   (그룹 추천 대상 사용자는 모두 과거 기록이 있으므로 이 조건이 실제 사용 상황과 같다)
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import load_movielens, movie_genre_table, rmse, mae, save_json, RESULTS, SEED  # noqa: E402
from models import MF, ContentModel, hybrid  # noqa: E402

r, _ = load_movielens()
tr = r[r.split == "train"]
va = r[(r.split == "validation") & r.user_id.isin(tr.user_id) & r.movie_id.isin(tr.movie_id)]
gt = movie_genre_table(r)
U, M, Y = tr.user_id.values, tr.movie_id.values, tr.rating_norm.values
vU, vM, vY = va.user_id.values, va.movie_id.values, va.rating_norm.values
print(f"Train {len(tr):,}건 / Validation 채점용 {len(va):,}건 (사용자 {va.user_id.nunique()}명)")

# ------------------------------------------------------------------ 1) MF
def run_mf(k, lr, reg):
    t = time.time()
    m = MF(k=k, lr=lr, reg=reg, epochs=60, seed=SEED).fit(U, M, Y, val=(vU, vM, vY), patience=3)
    p = m.predict(vU, vM)
    row = dict(k=k, lr=lr, reg=reg, best_epoch=m.best_epoch, val_rmse=rmse(vY, p), val_mae=mae(vY, p),
               seconds=round(time.time() - t, 1))
    print("  MF", row)
    return row, m, p

mf_rows = []
print("\n[MF] coarse search")
for k, lr, reg in itertools.product([8, 32, 128], [0.005, 0.02, 0.05], [0.005, 0.02, 0.08]):
    row, _, _ = run_mf(k, lr, reg)
    row["phase"] = "coarse"
    mf_rows.append(row)
best = min(mf_rows, key=lambda x: x["val_rmse"])
print("coarse best:", best)

print("\n[MF] fine search (best 주변)")
k0, lr0, reg0 = best["k"], best["lr"], best["reg"]
fine = set()
for k in [max(4, k0 // 2), k0, k0 * 2]:
    for reg in [reg0 / 2, reg0, reg0 * 2]:
        fine.add((k, lr0, round(reg, 5)))
for lr in [lr0 / 2, lr0 * 2]:
    fine.add((k0, round(lr, 5), reg0))
done = {(x["k"], x["lr"], x["reg"]) for x in mf_rows}
for k, lr, reg in sorted(fine - done):
    row, _, _ = run_mf(k, lr, reg)
    row["phase"] = "fine"
    mf_rows.append(row)
mf_df = pd.DataFrame(mf_rows).sort_values("val_rmse")
mf_df.to_csv(RESULTS / "stage1_mf_search.csv", index=False)
mf_best = mf_df.iloc[0].to_dict()
print("MF 최종:", mf_best)
_, mf_model, mf_val = run_mf(int(mf_best["k"]), mf_best["lr"], mf_best["reg"])

# ------------------------------------------------------------------ 2) Content
print("\n[Content] grid search")
c_rows = []
for k_mean, lam_user, lam_global in itertools.product([0, 2, 5, 10, 20], [1, 3, 10, 30, 100], [0.1, 1, 10]):
    c = ContentModel(k_mean, lam_user, lam_global).fit(U, M, Y, gt)
    p = c.predict(vU, vM)
    c_rows.append(dict(k_mean=k_mean, lam_user=lam_user, lam_global=lam_global, val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
c_df = pd.DataFrame(c_rows).sort_values("val_rmse")
c_df.to_csv(RESULTS / "stage1_content_search.csv", index=False)
c_best = c_df.iloc[0].to_dict()
print("Content 최종:", c_best)
content_model = ContentModel(c_best["k_mean"], c_best["lam_user"], c_best["lam_global"]).fit(U, M, Y, gt)
c_val = content_model.predict(vU, vM)

# ------------------------------------------------------------------ 3) Hybrid alpha
print("\n[Hybrid] alpha search")
h_rows = []
for a in np.round(np.arange(0, 1.0001, 0.05), 2):
    p = hybrid(mf_val, c_val, a)
    h_rows.append(dict(alpha=float(a), val_rmse=rmse(vY, p), val_mae=mae(vY, p)))
h_df = pd.DataFrame(h_rows)
h_df.to_csv(RESULTS / "stage1_hybrid_alpha.csv", index=False)
h_best = h_df.loc[h_df.val_rmse.idxmin()].to_dict()
print(h_df.to_string(index=False))

# ------------------------------------------------------------------ 4) 비교 & 선택
baseline_user_mean = np.array([content_model.user_mean[u] for u in vU])
compare = pd.DataFrame([
    dict(model="Baseline: 사용자 평균", val_rmse=rmse(vY, baseline_user_mean), val_mae=mae(vY, baseline_user_mean)),
    dict(model="MF", val_rmse=mf_best["val_rmse"], val_mae=mf_best["val_mae"]),
    dict(model="Content", val_rmse=c_best["val_rmse"], val_mae=c_best["val_mae"]),
    dict(model=f"Hybrid (alpha={h_best['alpha']:.2f})", val_rmse=h_best["val_rmse"], val_mae=h_best["val_mae"]),
])
compare.to_csv(RESULTS / "stage1_compare.csv", index=False)
print("\n", compare.to_string(index=False))

cands = {"MF": mf_best["val_rmse"], "Content": c_best["val_rmse"], "Hybrid": h_best["val_rmse"]}
chosen = min(cands, key=cands.get)
# Hybrid 가 alpha=1 또는 0 에서 최적이면 사실상 MF 또는 Content 와 같다 -> 그 이름으로 기록
if chosen == "Hybrid" and h_best["alpha"] in (0.0, 1.0):
    chosen = "MF" if h_best["alpha"] == 1.0 else "Content"
alpha = {"MF": 1.0, "Content": 0.0}.get(chosen, h_best["alpha"])
selection = dict(
    chosen_model=chosen, alpha=alpha,
    mf=dict(k=int(mf_best["k"]), lr=mf_best["lr"], reg=mf_best["reg"], epochs=int(mf_best["best_epoch"])),
    content=dict(k_mean=c_best["k_mean"], lam_user=c_best["lam_user"], lam_global=c_best["lam_global"]),
    validation_rmse=cands, validation_eval_rows=int(len(va)),
    rule="Validation RMSE 최소 모델 선택 (같은 평점 묶음으로 채점)",
)
save_json(selection, "stage1_selection.json")
print("\n선택:", selection)
