"""데모: 실제 추천처럼 4명에게 영화 1편 추천하기.

사용법: RESULTS_DIR=results_v1_frozen python 05_demo.py [group_id]   (기본값 1)

후보(Candidate Set) 만드는 법
  전체 MovieLens 영화
  -> 4명 중 누구라도 이미 평점을 남긴 영화 제외 (Train/Validation/Test 전체 기록 기준)
  -> 최종 모델이 점수를 낼 수 있는 영화만 유지 (MF 를 쓰는 구조면 MF가 학습한 영화만. 처음 보는 영화를 임의로 처리하지 않음)
  -> 신뢰도 필터: 학습 평점이 너무 적은 영화는 MF 좌표가 '찍은 값'에 가까움.
     Validation 오차를 영화별 학습 평점 수 구간으로 나눠 보면, 평점 20개 이하 영화는 오차가 크고 점수를 부풀린다.
     규칙: 'Validation RMSE 가 전체 평균의 1.1배 이내인 구간'만 남김 -> 기준 개수는 코드가 데이터에서 계산
  -> 모든 후보에 Final Score 계산 -> 1등 추천
※ 원본 데이터에 영화 제목이 없어 movie_id 와 장르로 표시한다 (movies.dat 를 붙이면 제목 표시 가능).
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, ROOT, SEED, load_json, load_movielens, movie_genre_table  # noqa: E402
from group import risk_proba  # noqa: E402
from models import Stage1Predictor  # noqa: E402
from risk import movie_stats, user_thresholds  # noqa: E402

gid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
final = load_json("FINAL_FROZEN_CONFIG.json")
with open(RESULTS / "risk_model.pkl", "rb") as f:
    risk = pickle.load(f)
r, groups = load_movielens()
trva = r[r.split.isin(["train", "validation"])]
gt = movie_genre_table(r)
s1 = Stage1Predictor(final["stage1"], gt, seed=SEED).fit(trva)

row = groups[groups.group_id == gid].iloc[0]
users = [int(row[f"user_id_{i}"]) for i in range(1, 5)]
seen = set(r[r.user_id.isin(users)].movie_id)
all_movies = sorted(set(trva.movie_id))
# 신뢰도 필터 기준: Train 만으로 학습한 모델의 Validation 오차(02 단계에서 저장)를 학습 평점 수 구간별로 본다
tr = r[r.split == "train"]
v = pd.read_csv(RESULTS / "val_stage1_predictions.csv")
v["n"] = v.movie_id.map(tr.groupby("movie_id").size())
EDGES = [0, 5, 10, 20, 50, 100, 200, 500, 10**9]
v["bucket"] = pd.cut(v.n, EDGES)
by = v.groupby("bucket", observed=True).apply(lambda d: np.sqrt(((d.rating_norm - d.pred) ** 2).mean()))
overall = np.sqrt(((v.rating_norm - v.pred) ** 2).mean())
bad = [b for b, e in by.items() if e > 1.1 * overall]
min_count = int(max(b.right for b in bad)) + 1 if bad else 1
print(f"신뢰도 필터: Validation 기준 학습 평점 {min_count}개 이상인 영화만 후보 (구간별 RMSE: "
      + ", ".join(f"{b.left}-{b.right if b.right < 10**9 else '+'}: {e:.3f}" for b, e in by.items()) + ")")
n_trva = trva.groupby("movie_id").size()
cand = [m for m in all_movies if m not in seen and s1.mf_knows_item([m])[0] and gt[m].sum() > 0
        and n_trva.get(m, 0) >= min_count]
print(f"그룹 {gid} 사용자 {users}: 이미 본 영화 {len(seen)}편 제외 -> 후보 {len(cand)}편")

pred = np.column_stack([s1.predict([u] * len(cand), cand) for u in users])
th = user_thresholds(trva, final["stage2"]["percentile"])
thr = np.tile(th.thr.loc[users].to_numpy(), (len(cand), 1))
share = np.tile(th.share.loc[users].to_numpy(), (len(cand), 1))
R = risk_proba(risk, pred, thr, share, np.array(cand), movie_stats(trva))
E = pred.mean(1)
# 최종 모델 = 1차 식. 'v4' 를 두 번째 인자로 주면 4차 실험 식(평균 + 0.5*최저 - 3.0*위험)으로 비교 가능.
v4 = ROOT / "results_v4" / "FINAL_FROZEN_CONFIG.json"
if len(sys.argv) > 2 and sys.argv[2] == "v4" and v4.exists():
    c4 = json.load(open(v4, encoding="utf-8"))
    F = E + c4["b"] * pred.min(1) - c4["a"] * R
    final["lam"] = f"{c4['a']} (+ 최저 예측 x {c4['b']})"
    tag = "_v4"   # 4차 비교 결과는 별도 파일로 저장 (1차 데모 결과를 덮어쓰지 않게)
else:
    F = E - final["lam"] * R
    tag = ""
genre_str = r.drop_duplicates("movie_id").set_index("movie_id").genres
titles = {}
if (ROOT / "data_extra" / "ml-1m" / "movies.dat").exists():   # 2차에 받은 movies.dat 이 있으면 제목 표시
    mv = pd.read_csv(ROOT / "data_extra" / "ml-1m" / "movies.dat", sep="::", engine="python",
                     names=["movie_id", "title", "g"], encoding="latin1")
    titles = dict(zip(mv.movie_id, mv.title))
out = pd.DataFrame({"movie_id": cand, "title": [titles.get(m, "") for m in cand],
                    "genres": [genre_str[m] for m in cand], "expected_satisfaction": E,
                    "misery_risk": R, "final_score": F,
                    **{f"pred_user{u}_stars": 1 + 4 * np.clip(pred[:, j], 0, 1) for j, u in enumerate(users)}})
out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
out.to_csv(RESULTS / f"demo_group{gid}{tag}.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nlambda = {final['lam']}  (Final = 예상만족 - lambda x Strong Dislike 위험)")
print(out.head(5).round(3).to_string())
avg_pick = out.loc[out.expected_satisfaction.idxmax()]
print(f"\n최종 추천: movie {int(out.movie_id[0])} {out.title[0]} ({out.genres[0]})  예상만족 {out.expected_satisfaction[0]:.3f}, "
      f"Strong Dislike 위험 {out.misery_risk[0]:.3f}")
print(f"참고 - 평균만 봤다면: movie {int(avg_pick.movie_id)} {avg_pick.title} ({avg_pick.genres})  예상만족 "
      f"{avg_pick.expected_satisfaction:.3f}, Strong Dislike 위험 {avg_pick.misery_risk:.3f}")
