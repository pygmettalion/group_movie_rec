"""Stage 2 도구: Strong Dislike 기준과 Group Risk(그룹 내 '극혐' 위험) 피처.

Strong Dislike (개인별 상대 기준)
  - 사람마다 별점 습관이 다르다. 짠돌이는 3점도 '별로', 후한 사람은 3점이면 '최악'일 수 있다.
  - 그래서 "그 사람이 과거에 준 점수들 중 하위 p% 선(threshold) 이하" 를 Strong Dislike 로 정의한다.
  - threshold 는 반드시 '예측 대상 기간보다 앞선 기록'으로만 계산한다 (정답 평점 섞임 = leakage 금지).

Group Misery Risk
  - "4명 중 적어도 한 명이 Strong Dislike 할 확률" 을 분류기로 예측한다.
  - 분류기 입력은 Stage 1 의 최종 개인 선호 점수(+사람별 기준선, 영화의 과거 평점 통계)뿐이다.
"""
import numpy as np
import pandas as pd

PERCENTILES = [5, 10, 15, 20, 25]


def user_thresholds(hist, p):
    """hist: 예측 시점 이전의 평점표. 반환: DataFrame(user_id -> thr, share)

    별점은 1,2,3,4,5 처럼 띄엄띄엄이라 같은 점수가 많다(동점).
    보통의 백분위 공식(q_p 이하)을 쓰면 '하위 5%' 라고 해 놓고 실제로는 그 사람 평점의 14% 가 걸린다(데이터로 확인).
    그래서 '그 점수 이하가 그 사람 평점의 p% 를 넘지 않는 가장 높은 점수' 를 극혐선으로 쓴다.
    (가장 낮은 점수 하나만으로도 p% 를 넘는 사람은 그 가장 낮은 점수를 극혐선으로 둔다.)
    share = 과거 평점 중 극혐선 이하 비율(그 사람이 '극혐'을 얼마나 자주 주는지).
    """
    cnt = hist.groupby(["user_id", "rating_norm"]).size().rename("n").reset_index()
    cnt["cum"] = cnt.groupby("user_id").n.cumsum() / cnt.groupby("user_id").n.transform("sum")
    ok = cnt[cnt.cum <= p / 100.0 + 1e-12].groupby("user_id").rating_norm.max()
    lowest = cnt.groupby("user_id").rating_norm.min()
    thr = ok.reindex(lowest.index).fillna(lowest)
    share = cnt.assign(thr=cnt.user_id.map(thr)).query("rating_norm <= thr").groupby("user_id").cum.max()
    return pd.DataFrame({"thr": thr, "share": share.reindex(thr.index)})


def movie_stats(hist):
    """영화별 과거 평점 통계 (모델 내부 표현이 아니라 '관측 데이터 요약'). 낮은 평점 비율 = 하위 25% 구간."""
    g = hist.groupby("movie_id").rating_norm
    st = pd.DataFrame({
        "m_logcount": np.log1p(g.size()),
        "m_mean": g.mean(),
        "m_std": g.std().fillna(0.0),
        "m_lowfrac": hist.assign(low=hist.rating_norm <= 0.25).groupby("movie_id").low.mean(),
    })
    return st


FEATURE_SETS = {
    "F1_pred": ["pred_mean", "pred_min", "pred_max", "pred_std"],
    "F2_pred_margin": ["pred_mean", "pred_min", "pred_max", "pred_std",
                       "margin_min", "margin_mean", "n_below", "share_max", "share_mean"],
    "F3_all": ["pred_mean", "pred_min", "pred_max", "pred_std",
               "margin_min", "margin_mean", "n_below", "share_max", "share_mean",
               "m_logcount", "m_mean", "m_std", "m_lowfrac"],
}


def group_features(pred, thr, share, movie_ids, mstats):
    """pred/thr/share: (그룹-영화 수, 4) 배열. margin = 예측점수 - 그 사람의 극혐선 (음수면 위험)."""
    pred, thr, share = map(lambda a: np.asarray(a, float), (pred, thr, share))
    margin = pred - thr
    f = pd.DataFrame({
        "pred_mean": pred.mean(1), "pred_min": pred.min(1), "pred_max": pred.max(1), "pred_std": pred.std(1),
        "margin_min": margin.min(1), "margin_mean": margin.mean(1), "n_below": (margin < 0).sum(1),
        "share_max": share.max(1), "share_mean": share.mean(1),
    })
    ms = mstats.reindex(movie_ids)
    global_fill = mstats.mean()
    for c in ["m_logcount", "m_mean", "m_std", "m_lowfrac"]:
        f[c] = ms[c].fillna(0.0 if c == "m_logcount" else global_fill[c]).to_numpy()
    return f


# 개인 단위 위험 모델용 (추가 후보 구조): 한 사람-한 영화마다 '극혐 확률'을 구한다.
INDIV_FEATURES = ["pred", "thr", "margin", "share", "m_logcount", "m_mean", "m_std", "m_lowfrac"]


def indiv_features(pred, thr, share, movie_ids, mstats):
    pred, thr, share = (np.asarray(a, float) for a in (pred, thr, share))
    f = pd.DataFrame({"pred": pred, "thr": thr, "margin": pred - thr, "share": share})
    ms = mstats.reindex(movie_ids)
    global_fill = mstats.mean()
    for c in ["m_logcount", "m_mean", "m_std", "m_lowfrac"]:
        f[c] = ms[c].fillna(0.0 if c == "m_logcount" else global_fill[c]).to_numpy()
    return f
