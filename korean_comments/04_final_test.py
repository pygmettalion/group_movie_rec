"""Stage 4: 고정된 모델로 최종 시험 (한 번만 채점, 이 결과로 아무것도 다시 고르지 않는다).

 1) Internal Test   : MovieLens Test 기간, 공식 그룹 49개
 2) External Test   : MovieTweetings 44개 그룹 (전혀 다른 서비스의 사용자들)
 3) Beliefs 검증    : 아직 안 본 영화에 대한 '관람 전 예상 별점'을 개인 선호 모델이 얼마나 맞히는지

모든 설정은 results/FINAL_FROZEN_CONFIG.json 에서 읽는다.
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import (RESULTS, SEED, group_rows, load_beliefs, load_json, load_movielens, load_movietweetings,  # noqa
                    mae, movie_genre_table, rmse, save_json)
from group import build_case, evaluate, paired_bootstrap, summarize  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

final = load_json("FINAL_FROZEN_CONFIG.json")
S1, P, LAM = final["stage1"], final["stage2"]["percentile"], final["lam"]
with open(RESULTS / "risk_model.pkl", "rb") as f:
    RISK = pickle.load(f)
print(f"고정 설정: Stage1={S1['structure']}, percentile={P}, risk={final['stage2']['structure']}/"
      f"{final['stage2']['classifier']}, lambda={LAM}")

r, groups = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
te = r[r.split == "test"]
gt_ml = movie_genre_table(r)

# 고정된 하이퍼파라미터로 Train+Validation 재학습 (공부 범위만 시험 직전까지 확장)
s1 = Stage1Predictor(S1, gt_ml, seed=SEED).fit(trva)
mstats = movie_stats(trva)
METHODS = ["Random", "Average", "LeastMisery", "RiskAware", "Oracle"]


def run_groups(cases, tag):
    per_group, summary = [], []
    for m in METHODS:
        lam = LAM if m == "RiskAware" else 0.0
        df = evaluate(cases, m, lam)
        per_group.append(df)
        summary.append(dict(method=m if m != "RiskAware" else f"RiskAware(lambda={LAM})", **summarize(df)))
    per_group = pd.concat(per_group)
    per_group.to_csv(RESULTS / f"test_{tag}_per_group.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / f"test_{tag}_methods.csv", index=False)
    g = {m: per_group[per_group.method == m].sort_values("group_id").reset_index(drop=True) for m in METHODS}
    boot = {f"RiskAware - {b}": paired_bootstrap(g["RiskAware"], g[b]) for b in ["Random", "Average", "LeastMisery"]}
    print(f"\n[{tag}] 그룹 {len(cases)}개\n", summary.round(4).to_string(index=False))
    for k, v in boot.items():
        print(f"  {k}: " + ", ".join(f"{m} {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for m, d in v.items()))
    return summary, boot


# ================================================================== 1) Internal Test
# (참고) 개인 선호 예측 정확도 - 보고용. 선택에는 쓰지 않음.
te_w = te[te.user_id.isin(trva.user_id) & te.movie_id.isin(trva.movie_id)]
ind = [dict(model=f"최종 Stage 1 ({S1['structure']})", test_rmse=rmse(te_w.rating_norm, s1.predict(te_w.user_id.values, te_w.movie_id.values)))]
if s1.use_mf:
    ind.append(dict(model="MF component", test_rmse=rmse(te_w.rating_norm, s1.mf.predict(te_w.user_id.values, te_w.movie_id.values))))
if s1.use_content:
    ind.append(dict(model="Content component", test_rmse=rmse(te_w.rating_norm, s1.content.predict(te_w.user_id.values, te_w.movie_id.values))))
ind_df = pd.DataFrame(ind)
ind_df["test_rmse_stars"] = ind_df.test_rmse * 4
ind_df.to_csv(RESULTS / "test_internal_individual.csv", index=False)
print("\n[Internal] 개인 예측 RMSE (Test 평점", len(te_w), "건)\n", ind_df.round(4).to_string(index=False))

th = user_thresholds(trva, P)
truth = te.set_index(["user_id", "movie_id"]).rating_norm
cases = []
for gid, users, movies in group_rows(groups, "test"):
    pred = np.column_stack([s1.predict([u] * len(movies), movies) for u in users])
    true = np.array([[truth[(u, m)] for u in users] for m in movies])
    cases.append(build_case(gid, users, movies, pred, true, th.thr.loc[users].to_numpy(), th.share.loc[users].to_numpy(),
                            RISK, mstats))
int_sum, int_boot = run_groups(cases, "internal")

# ================================================================== 2) External Test (MovieTweetings)
mt, mt_groups = load_movietweetings()
hist, mt_test = mt[mt.split == "history"], mt[mt.split == "test"]
gt_mt = movie_genre_table(mt)                      # MovieTweetings(IMDb) 장르 - 사용자 프로필과 같은 어휘
gt_mt_int = {int(k) if k.isdigit() else k: v for k, v in gt_mt.items()}


def as_ml_id(m):
    return int(m) if m.isdigit() else m


th_mt = user_thresholds(hist, P)
truth_mt = mt_test.assign(mid=mt_test.movie_id.astype(int)).set_index(["user_id", "mid"]).rating_norm
hist_by_user = {u: d for u, d in hist.groupby("user_id")}
cases_mt, coverage = [], []
for gid, users, movies in group_rows(mt_groups, "test"):
    cols = []
    for u in users:
        h = hist_by_user[u]
        p_u, known = s1.predict_new_user([as_ml_id(m) for m in h.movie_id], h.rating_norm.values, movies,
                                         genre_table=gt_mt_int, allow_content_fallback=False)
        coverage.append(known.mean())
        cols.append(p_u)
    pred = np.column_stack(cols)
    true = np.array([[truth_mt[(u, m)] for u in users] for m in movies])
    cases_mt.append(build_case(gid, users, movies, pred, true, th_mt.thr.loc[users].to_numpy(),
                               th_mt.share.loc[users].to_numpy(), RISK, mstats))
print(f"\n[External] MF fold-in 가능 비율: {np.mean(coverage):.3f} (1.0 이면 모든 사용자·후보에 대해 Hybrid 점수 생성)")
assert not np.isnan(np.concatenate([c["pred"].ravel() for c in cases_mt])).any(), "예측 불가 후보 존재"
ext_sum, ext_boot = run_groups(cases_mt, "external")

# ================================================================== 3) Beliefs (미시청 영화 개인 선호)
b_test, b_hist = load_beliefs()
gt_b = movie_genre_table(b_hist, b_test)           # 영화 메타데이터(장르)만 사용, 평점은 아님
b_hist = b_hist.sort_values(["user_id", "timestamp"])
hist_groups = {u: (d.timestamp.to_numpy(), d.movie_id.to_numpy(), d.rating_norm.to_numpy()) for u, d in b_hist.groupby("user_id")}
out = []
for (u, ts), d in b_test.groupby(["user_id", "timestamp"]):
    hts, hm, hy = hist_groups.get(u, (np.array([]), np.array([]), np.array([])))
    cut = np.searchsorted(hts, ts, side="left")          # 응답 시각보다 '앞선' 평점만
    hm_, hy_ = hm[:cut], hy[:cut]
    movies = d.movie_id.to_numpy()
    pred, known = s1.predict_new_user(hm_, hy_, movies, genre_table=gt_b, allow_content_fallback=True)
    base = (hy_.mean() if len(hy_) else np.nan)
    out.append(pd.DataFrame({"user_id": u, "movie_id": movies, "y": d.rating_norm.values, "pred": pred,
                             "mf_covered": known, "user_mean": base, "n_hist": cut}))
bel = pd.concat(out, ignore_index=True)
bel.to_csv(RESULTS / "beliefs_predictions.csv", index=False)
STAR = 4.5   # Beliefs 0.5~5점 -> 0~1 로 바꿀 때 나눈 값
rows = []
for name, mask in [("전체", np.ones(len(bel), bool)), ("MF가 아는 영화 (Hybrid 그대로)", bel.mf_covered.to_numpy()),
                   ("MF가 모르는 영화 (Content 대체)", ~bel.mf_covered.to_numpy())]:
    d = bel[mask & bel.user_mean.notna()]
    rows.append(dict(subset=name, n=len(d), model_rmse_stars=rmse(d.y, d.pred) * STAR, model_mae_stars=mae(d.y, d.pred) * STAR,
                     user_mean_rmse_stars=rmse(d.y, d.user_mean) * STAR, corr=float(np.corrcoef(d.y, d.pred)[0, 1])))
bel_df = pd.DataFrame(rows)
bel_df.to_csv(RESULTS / "beliefs_individual.csv", index=False)
print("\n[Beliefs] 관람 전 예상 별점 맞히기 (별점 단위 RMSE)\n", bel_df.round(4).to_string(index=False))

save_json(dict(internal=int_sum.to_dict("records"), internal_bootstrap=int_boot,
               external=ext_sum.to_dict("records"), external_bootstrap=ext_boot,
               external_mf_coverage=float(np.mean(coverage)),
               internal_individual=ind_df.to_dict("records"), beliefs=bel_df.to_dict("records")), "FINAL_TEST_RESULTS.json")
print("\n저장 완료 -> results/FINAL_TEST_RESULTS.json")
