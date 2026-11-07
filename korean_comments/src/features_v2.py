"""2차 버전 영화 피처: 장르 + 개봉 연도 + 평론가 점수 + 감독 효과.

Content 모델은 '영화마다 숫자 벡터 하나'만 있으면 되므로, 장르 벡터 뒤에 새 정보를 이어 붙인 표를 만든다.
  G : 장르 18칸 (1차와 같음)
  Y : 개봉 연도 (1985년=0, 15년=1 단위)
  C : 평론가 점수 (있으면 표준화 값, 없으면 0) + '평론가 점수 없음' 표시
  D : 감독 효과 = 같은 감독의 '다른 영화'들이 평균보다 얼마나 높게/낮게 평가받았는지
      (그 영화 자기 자신의 평점은 빼고 계산 = leave-one-movie-out, 정답 섞임 방지)
      과거 기록(hist)만으로 계산하고, 평점이 적은 감독은 0 쪽으로 당긴다(kappa).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import genre_vector
from metadata import build_metadata

FEATURE_SETS = {"G": [], "GY": ["year"], "GYC": ["year", "critic"], "GYCD": ["year", "critic", "director"]}
_META = None


def meta():
    global _META
    if _META is None:
        m = build_metadata()
        _META = m.set_index("movie_id")
    return _META


def director_effects(hist, kappa):
    """감독별 (평점 - 그 사용자 평균) 합계와 개수. 영화별로 따로 저장해서 '자기 영화 빼기'가 가능하게."""
    md = meta()
    um = hist.groupby("user_id").rating_norm.transform("mean")
    res = (hist.rating_norm - um).groupby(hist.movie_id).agg(["sum", "count"])
    d_sum, d_cnt, per_movie = defaultdict(float), defaultdict(float), {}
    for m, (s, c) in res.iterrows():
        dirs = md.directors.get(m) if m in md.index else None
        if isinstance(dirs, list):
            per_movie[m] = (dirs, s, c)
            for d in dirs:
                d_sum[d] += s
                d_cnt[d] += c
    out = {}
    for m in md.index:
        dirs = md.directors.get(m)
        if not isinstance(dirs, list):
            continue
        own = per_movie.get(m)
        vals = []
        for d in dirs:
            s, c = d_sum.get(d, 0.0), d_cnt.get(d, 0.0)
            if own:
                s, c = s - own[1], c - own[2]
            vals.append(s / (c + kappa))
        out[m] = float(np.mean(vals))
    return out


def extra_vector(m, parts, deff):
    md = meta()
    v = []
    known = m in md.index
    if "year" in parts:
        v.append((md.year.get(m) - 1985) / 15.0 if known else 0.0)
    if "critic" in parts:
        c = md.critic.get(m) if known else np.nan
        v += [0.0, 1.0] if pd.isna(c) else [(c - 0.6) / 0.25, 0.0]
    if "director" in parts:
        v.append(deff.get(m, 0.0))
    return np.array(v, float)


def build_table(hist, genre_table, feature_set, kappa=50.0, extra_ids=()):
    """movie_id -> [장르 | 추가 피처] 벡터 표. hist 는 '예측 시점 이전' 평점만."""
    parts = FEATURE_SETS[feature_set]
    deff = director_effects(hist, kappa) if "director" in parts else {}
    ids = set(genre_table) | set(meta().index) | set(extra_ids)
    zero_g = np.zeros(18)
    return {m: np.concatenate([genre_table.get(m, zero_g), extra_vector(m, parts, deff)]) for m in ids}


def extend_with_genres(table_v2, other_genre_table, feature_set):
    """다른 데이터셋(MovieTweetings/Beliefs)의 장르 표기를 쓰되, 추가 피처는 MovieLens 번호가 있으면 가져온다."""
    n_extra = len(next(iter(table_v2.values()))) - 18
    out = {}
    for m, g in other_genre_table.items():
        base = table_v2.get(m)
        extra = base[18:] if base is not None else _missing_extra(feature_set)
        out[m] = np.concatenate([g, extra]) if n_extra else g
    return out


def _missing_extra(feature_set):
    parts = FEATURE_SETS[feature_set]
    v = []
    if "year" in parts:
        v.append(0.0)
    if "critic" in parts:
        v += [0.0, 1.0]
    if "director" in parts:
        v.append(0.0)
    return np.array(v, float)


def risk_item_table(hist, kappa=50.0):
    """위험 모델용 영화 정보: 평론가 점수(없으면 표시), 개봉 연도, 감독 효과."""
    md = meta()
    deff = director_effects(hist, kappa)
    df = pd.DataFrame({"critic": md.critic, "year": md.year})
    df["critic_missing"] = df.critic.isna().astype(float)
    df["critic"] = df.critic.fillna(df.critic.mean())
    df["year_z"] = (df.year - 1985) / 15.0
    df["dir_effect"] = pd.Series(deff).reindex(df.index).fillna(0.0)
    return df[["critic", "critic_missing", "year_z", "dir_effect"]]


def user_genre_dislike(hist, thr, alpha):
    """사용자별 '장르별 Strong Dislike 비율' (18칸). 기록이 적은 장르는 그 사람 전체 Strong Dislike 비율 쪽으로 당김(alpha)."""
    h = hist[["user_id", "movie_id", "rating_norm", "genres"]].copy()
    h["dis"] = (h.rating_norm <= h.user_id.map(thr)).astype(float)
    G = np.vstack([(genre_vector(g) > 0).astype(float) for g in h.genres])
    out = {}
    codes, uniq = pd.factorize(h.user_id)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    dis = h.dis.to_numpy()
    for j, u in enumerate(uniq):
        rows = order[bounds[j]:bounds[j + 1]]
        base = dis[rows].mean()
        n = G[rows].sum(0)
        k = (G[rows] * dis[rows, None]).sum(0)
        out[u] = (k + alpha * base) / (n + alpha)
    return out
