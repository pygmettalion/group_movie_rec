"""Stage 3: 위험 벌점 크기 lambda 고르기 -> 전체 모델 고정(freeze).

공식 Validation 그룹 49개 + 추가 연습 그룹(02 에서 교차 적합으로 만든 것)에서, 각 그룹의 Validation 후보 영화
(4명 모두 실제 평점 있음) 중 모델이 1편을 고르면 그때 실제 평점을 공개해서 채점한다.
공식 그룹만 쓴 결과도 함께 저장해서 두 결과가 비슷한 방향인지 확인한다.

lambda 선택 규칙 (결과 보기 전에 정함)
  balanced_score = (평균 만족도 + 최저 만족도 + (1 - Strong Dislike 발생률)) / 3 이 최대인 lambda
  -> 세 가지 목표를 똑같은 비중으로 본다. 동점이면 더 작은 lambda (덜 과감한 쪽).
"""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, group_rows, load_json, load_movielens, save_json  # noqa: E402
from group import LAMBDA_GRID, build_case, evaluate, summarize  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

r, groups = load_movielens()
tr = r[r.split == "train"]
va = r[r.split == "validation"]
s1sel, s2sel = load_json("stage1_selection_final.json"), load_json("stage2_selection.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
p = s2sel["percentile"]

preds = pd.read_csv(RESULTS / "val_stage1_predictions.csv").set_index(["user_id", "movie_id"]).pred
truth = va.set_index(["user_id", "movie_id"]).rating_norm
th = user_thresholds(tr, p)
mstats = movie_stats(tr)

cases = []
for gid, users, movies in group_rows(groups, "validation"):
    pred = np.array([[preds[(u, m)] for u in users] for m in movies])
    true = np.array([[truth[(u, m)] for u in users] for m in movies])
    cases.append(build_case(gid, users, movies, pred, true, th.thr.loc[users].to_numpy(),
                            th.share.loc[users].to_numpy(), risk, mstats))
print(f"Validation 그룹 {len(cases)}개, 후보 영화 평균 {np.mean([len(c['movies']) for c in cases]):.1f}편")

# 추가 연습 그룹 (위험 점수는 그 사람들을 학습에 안 쓴 분류기로 이미 계산됨)
extra = pd.read_csv(RESULTS / "stage2_extra_val_groups.csv")
extra_cases = []
for gid_, d in extra.groupby("group_id"):
    pred = d[["p1", "p2", "p3", "p4"]].to_numpy()
    extra_cases.append(dict(group_id=gid_, users=d.users.iloc[0].split("|"), movies=d.movie_id.tolist(), pred=pred,
                            true=d[["y1", "y2", "y3", "y4"]].to_numpy(), thr=d[["thr1", "thr2", "thr3", "thr4"]].iloc[0].to_numpy(),
                            exp_sat=pred.mean(1), risk=d.risk.to_numpy()))
all_cases = cases + extra_cases
print(f"추가 연습 그룹 {len(extra_cases)}개 -> lambda 선택에 총 {len(all_cases)}개 그룹 사용")

rows = []
for lam in LAMBDA_GRID:
    s_all = summarize(evaluate(all_cases, "RiskAware", lam))
    s_off = summarize(evaluate(cases, "RiskAware", lam))
    rows.append(dict(lam=lam, **s_all, **{f"official_{k}": v for k, v in s_off.items()}))
lam_df = pd.DataFrame(rows)
lam_df.to_csv(RESULTS / "stage3_lambda_search.csv", index=False)
print(lam_df.round(4).to_string(index=False))

best_score = lam_df.balanced_score.max()
lam_star = float(lam_df[lam_df.balanced_score >= best_score - 1e-12].lam.min())
print(f"\n선택된 lambda = {lam_star}")

base = []
for name, cs in [("all", all_cases), ("official49", cases)]:
    for m in ["Random", "Average", "LeastMisery", "Oracle"]:
        base.append(dict(groups=name, method=m, **summarize(evaluate(cs, m))))
    base.append(dict(groups=name, method=f"RiskAware(lambda={lam_star})", **summarize(evaluate(cs, "RiskAware", lam_star))))
base_df = pd.DataFrame(base)
base_df.to_csv(RESULTS / "stage3_validation_methods.csv", index=False)
print("\n", base_df.round(4).to_string(index=False))

final = dict(
    frozen_note="이 파일이 저장된 뒤에는 어떤 값도 바꾸지 않는다. Test/External Test 는 이 설정으로 한 번만 채점.",
    stage1=s1sel, stage2=s2sel, lam=lam_star,
    lambda_rule="공식49+추가 연습 그룹 전체에서 balanced_score=(평균만족+최저만족+(1-극혐률))/3 최대, 동점이면 작은 lambda",
    n_lambda_groups=len(all_cases),
    test_refit_note="Test 채점 전, 고정된 하이퍼파라미터 그대로 Stage 1 을 Train+Validation 으로 다시 학습 "
                    "(설정은 그대로, 시험 직전까지의 기록으로 공부 범위만 늘림). Risk 분류기와 lambda 는 그대로.",
)
save_json(final, "FINAL_FROZEN_CONFIG.json")
print("\n모델 고정 완료 -> results/FINAL_FROZEN_CONFIG.json")
