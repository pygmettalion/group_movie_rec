"""[2차 버전] Strong Dislike 위험 모델 피처 추가.

1차 피처(개인 예측 점수, 불호 기준선, 거리, Strong Dislike 빈도, 영화 과거 평점 통계)에 더해
  - meta : 평론가 점수(+없음 표시), 개봉 연도, 감독 효과      -> "평론가가 혹평한 영화는 누군가 강하게 싫어하기 쉽다?"
  - gd   : 그 사람의 '이 영화 장르에서의 과거 Strong Dislike 비율'       -> "공포 영화만 나오면 1점 주는 사람"
모두 예측 시점 이전 기록(hist)만으로 계산한다.
"""
import numpy as np

from common import genre_vector
from features_v2 import risk_item_table, user_genre_dislike
from risk import INDIV_FEATURES, indiv_features

META_COLS = ["critic", "critic_missing", "year_z", "dir_effect"]
INDIV_V2_SETS = {
    "I_member": INDIV_FEATURES,
    "I_meta": INDIV_FEATURES + META_COLS,
    "I_gd": INDIV_FEATURES + ["u_gdis"],
    "I_meta_gd": INDIV_FEATURES + META_COLS + ["u_gdis"],
}


class RiskContext:
    def __init__(self, gd_hist, thr, genre_lookup, alpha, item_hist, kappa):
        """gd_hist: 그 사용자들의 예측 시점 이전 평점(genres 열 포함) -> 장르별 Strong Dislike 비율
        item_hist: 영화 정보(감독 효과)를 계산할 MovieLens 과거 평점, thr: user->불호 기준선,
        genre_lookup: movie->장르 문자열(MovieLens 표기)."""
        self.items = risk_item_table(item_hist, kappa)
        self.gdis = user_genre_dislike(gd_hist, thr, alpha)
        self.genre_lookup = genre_lookup
        self._gcache = {}

    def gvec(self, m):
        if m not in self._gcache:
            v = (genre_vector(self.genre_lookup.get(m)) > 0).astype(float)
            self._gcache[m] = v / v.sum() if v.sum() > 0 else v
        return self._gcache[m]


def indiv_features_v2(pred, thr, share, movie_ids, user_ids, mstats, ctx, item_table=None):
    f = indiv_features(pred, thr, share, movie_ids, mstats)
    it = (item_table if item_table is not None else ctx.items).reindex(movie_ids)
    f["critic"] = it.critic.fillna(ctx.items.critic.mean()).to_numpy()
    f["critic_missing"] = it.critic_missing.fillna(1.0).to_numpy()
    f["year_z"] = it.year_z.fillna(0.0).to_numpy()
    f["dir_effect"] = it.dir_effect.fillna(0.0).to_numpy()
    f["u_gdis"] = [float(ctx.gvec(m) @ ctx.gdis[u]) if u in ctx.gdis else np.nan for u, m in zip(user_ids, movie_ids)]
    f["u_gdis"] = f.u_gdis.fillna(f.share)
    return f


def risk_proba_v2(risk, pred, thr, share, movie_ids, users, mstats, ctx, item_table=None):
    """pred/thr/share: (후보수, 4). users: 4명. 개인 확률 p_i -> 1 - prod(1 - p_i)."""
    cols = []
    for j in range(pred.shape[1]):
        X = indiv_features_v2(pred[:, j], thr[:, j], share[:, j], movie_ids, [users[j]] * len(movie_ids),
                              mstats, ctx, item_table)
        cols.append(risk["model"].predict_proba(X[risk["features"]])[:, 1])
    return 1 - np.prod(1 - np.column_stack(cols), axis=1)
