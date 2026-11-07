"""공통 도구: 데이터 읽기, 장르 벡터, 채점 함수.

쉽게 말하면: 모든 단계가 같이 쓰는 '공용 서랍'이다.
- 평점은 전부 0~1 사이로 맞춘 rating_norm 을 쓴다 (데이터셋마다 별점 범위가 달라서).
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))  # 2차 버전은 RESULTS_DIR=results_v2
RESULTS.mkdir(exist_ok=True)

GENRES = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']
G_INDEX = {g: i for i, g in enumerate(GENRES)}

SEED = 42


def read_csv(rel):
    return pd.read_csv(DATA / rel, encoding="utf-8-sig")


def load_movielens():
    r = read_csv("01_MovieLens_1M/ratings.csv")
    groups = read_csv("01_MovieLens_1M/eval_groups.csv")
    return r, groups


def load_movietweetings():
    r = read_csv("02_MovieTweetings/movietweetings_ratings.csv")
    r["movie_id"] = r["movie_id"].astype(str)
    groups = read_csv("02_MovieTweetings/movietweetings_test_groups.csv")
    return r, groups


def load_beliefs():
    test = read_csv("03_Beliefs/unseen_test.csv")
    hist = read_csv("03_Beliefs/past_history.csv")
    return test, hist


def genre_vector(genres):
    """장르 문자열 'Action|Comedy' -> 18칸짜리 벡터. 합이 1이 되도록 나눈다(장르 '구성비')."""
    v = np.zeros(len(GENRES))
    if isinstance(genres, str):
        for g in genres.split("|"):
            if g in G_INDEX:
                v[G_INDEX[g]] = 1.0
    s = v.sum()
    return v / s if s > 0 else v


def movie_genre_table(*frames):
    """여러 평점표에서 movie_id -> 장르벡터 사전을 만든다."""
    table = {}
    for f in frames:
        for mid, g in f[["movie_id", "genres"]].drop_duplicates("movie_id").itertuples(index=False):
            if mid not in table or (isinstance(g, str) and table[mid].sum() == 0):
                table[mid] = genre_vector(g)
    return table


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def save_json(obj, name):
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def group_rows(groups, split):
    """그룹표에서 (group_id, [4명], [후보 영화들]) 목록을 꺼낸다."""
    out = []
    col = f"{split}_movie_ids"
    for row in groups.itertuples(index=False):
        movies = getattr(row, col)
        if not isinstance(movies, str) or not movies:
            continue
        users = [getattr(row, f"user_id_{i}") for i in range(1, 5)]
        out.append((row.group_id, users, [int(m) for m in movies.split("|")]))
    return out
