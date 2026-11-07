"""[추가 실험] MovieLens 10M 준비: 1M 과 '똑같은 규칙'으로 데이터와 시험 그룹을 만든다.

왜 하나?
  1M 에서는 '4명이 공통으로 본 영화가 많은 그룹'이 드물어서, 후보가 많을 때의 실력을 크게 보여주기 어려웠다.
  10M(사용자 약 7만 명)으로 같은 방법을 처음부터 따로 적용해 본다. 1M 에서 정한 숫자는 하나도 바꾸지 않는다.

규칙 (팀 전처리 코드 prepare_team_data.py 와 같음)
  - 시간 순서 분할: 평점 시각 기준 앞 70% = Train, 다음 15% = Validation, 마지막 15% = Test
  - 그룹 사용자: Train+Validation 평점 10개 이상
  - 평점 정규화: 0.5~5점 -> (점수-0.5)/4.5 (0~1)
  - Validation 그룹: Validation 기간 공통 영화 5편 이상 (위험 모델 학습용 가짜 그룹은 따로)
  - Test 그룹: Test 기간 공통 영화 5편 이상, 그리고 10편 이상 묶음을 따로 표시 (별점은 보지 않고 봉인)

입력: data_extra/ml-10m/ (ratings.dat, movies.dat)   출력: results_ml10m/
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_ml10m"
OUT.mkdir(exist_ok=True)
SEED, MIN_HISTORY, MIN_COMMON, N_GROUPS = 42, 10, 5, 1500


def load(src):
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        rname = next(n for n in z.namelist() if n.endswith("ratings.dat"))
        mname = next(n for n in z.namelist() if n.endswith("movies.dat"))
        rf, mf = z.open(rname), z.open(mname)
    else:
        rf, mf = open(src / "ratings.dat", "rb"), open(src / "movies.dat", "rb")
    r = pd.read_csv(rf, sep="::", engine="python", names=["user_id", "movie_id", "rating_raw", "timestamp"])
    m = pd.read_csv(mf, sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="latin1")
    return r, m


src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_extra" / "ml-10m"
r, movies = load(src)
r["rating_norm"] = (r.rating_raw - 0.5) / 4.5
movies["genres"] = movies.genres.str.replace("Children", "Children's", regex=False)   # 1M 표기와 맞춤
r = r.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
ts = np.sort(r.timestamp.to_numpy())
v_start, t_start = ts[len(ts) * 70 // 100], ts[len(ts) * 85 // 100]
r["split"] = np.where(r.timestamp < v_start, "train", np.where(r.timestamp < t_start, "validation", "test"))
r.to_pickle(OUT / "ratings.pkl")
movies.to_pickle(OUT / "movies.pkl")
print(f"평점 {len(r):,} / 사용자 {r.user_id.nunique():,} / 영화 {r.movie_id.nunique():,}")
print(r.split.value_counts().to_dict(), "| Validation 시작", pd.to_datetime(v_start, unit="s"), "Test 시작", pd.to_datetime(t_start, unit="s"))


MAX_PER_USER = 10   # 한 사람이 들어갈 수 있는 그룹 수 상한 (영화 많이 본 소수가 시험을 독차지하지 않게)


def find_groups(period, exclude_users, n_max, rng, min_common):
    """period 기간에 4명이 공통으로 본 영화가 min_common 편 이상인 그룹 (별점은 쓰지 않음).
    사람을 무작위 순서로 돌며 '같이 본 영화가 많은 사람'을 찾아 4명을 묶고, 한 사람당 그룹 수를 MAX_PER_USER 로 제한."""
    from collections import Counter
    hist = r[r.timestamp < (v_start if period == "validation" else t_start)].groupby("user_id").size()
    ok = set(hist[hist >= MIN_HISTORY].index) - exclude_users
    known = set(r[r.split == "train"].movie_id) if period == "validation" else set(r[r.split != "test"].movie_id)
    p = r[(r.split == period) & r.user_id.isin(ok) & r.movie_id.isin(known)][["user_id", "movie_id"]]
    sets = {u: set(d.movie_id) for u, d in p.groupby("user_id") if len(d) >= min_common}
    by_movie = {}
    for u, ms in sets.items():
        for m in ms:
            by_movie.setdefault(m, []).append(u)
    used, found = Counter(), set()
    seeds = list(sets)
    rng.shuffle(seeds)
    for a in seeds:
        if len(found) >= n_max:
            break
        if used[a] >= MAX_PER_USER:
            continue
        co = Counter()
        for m in sets[a]:
            co.update(by_movie[m])
        partners = [b for b, c in co.items() if b != a and c >= min_common and used[b] < MAX_PER_USER]
        if len(partners) < 3:
            continue
        for _ in range(30):
            trio = tuple(rng.choice(partners, 3, replace=False))
            g = tuple(sorted((a, *trio)))
            if g in found or any(used[x] >= MAX_PER_USER for x in g):
                continue
            if len(sets[a] & sets[trio[0]] & sets[trio[1]] & sets[trio[2]]) >= min_common:
                found.add(g)
                used.update(g)
                break
    found = sorted(found)
    rows = []
    for gid, g in enumerate(found, 1):
        common = sorted(set.intersection(*[sets[u] for u in g]))
        rows.append(dict(group_id=gid, user_id_1=g[0], user_id_2=g[1], user_id_3=g[2], user_id_4=g[3],
                         n_common=len(common), movie_ids="|".join(map(str, common))))
    return pd.DataFrame(rows), len(sets)


rng = np.random.default_rng(SEED)
test_groups, n_found = find_groups("test", set(), N_GROUPS, rng, MIN_COMMON)
test_groups.to_csv(OUT / "TEST_groups_sealed.csv", index=False)
print(f"Test 그룹 봉인: 후보 사용자 {n_found:,}명, 그룹 {len(test_groups)}개, 후보 평균 {test_groups.n_common.mean():.1f}편, "
      f"10편 이상 {int((test_groups.n_common >= 10).sum())}개, 사용자 {len(set(test_groups.iloc[:, 1:5].values.ravel()))}명")

# 후보가 많은 그룹(공통 10편 이상)도 따로 봉인 — '고를 영화가 많을 때' 실력 확인용
groups10, n10 = find_groups("test", set(), N_GROUPS, np.random.default_rng(SEED + 1), 10)
groups10.to_csv(OUT / "TEST_groups10_sealed.csv", index=False)
print(f"Test 그룹(10편 이상) 봉인: 후보 사용자 {n10:,}명, 그룹 {len(groups10)}개, 후보 평균 {groups10.n_common.mean():.1f}편, "
      f"사용자 {len(set(groups10.iloc[:, 1:5].values.ravel()))}명")
