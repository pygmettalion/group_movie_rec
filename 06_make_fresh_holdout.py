"""2차 버전용 '새 시험지(Fresh Holdout)' 만들기 — 정답은 보지 않고 봉인만 한다.

왜 필요한가?
  1차 버전의 Internal/External Test 점수는 이미 봤다. 이후에 모델을 고치면 그 시험지로는 공정한 비교가 안 된다.
  그래서 '아직 아무도 채점해 본 적 없는' 그룹을 Test 기간에서 새로 만들어 봉인한다.
  - 공식 49그룹의 사용자 36명은 제외
  - 조건은 공식 그룹과 같게: 4명이 Test 기간에 공통으로 본 영화 5편 이상, 각자 Train+Validation 기록 10개 이상
  - 이 스크립트는 '누가 어떤 영화를 봤는지'만 보고, 별점(rating)은 열어 보지 않는다.
  - 1차 버전과 2차 버전을 이 시험지로 '딱 한 번씩' 채점해서 비교한다.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_movielens, save_json  # noqa: E402

MIN_COMMON = int(os.environ.get("MIN_COMMON", 5))      # 3번 시험지는 남은 사용자 기록이 적어 3 사용
MIN_HISTORY = 10
MAX_GROUPS = int(os.environ.get("MAX_GROUPS", 500))
# 사용법: python 06_make_fresh_holdout.py [이름] [제외할_이전_시험지.csv ...]
#   3차용: python 06_make_fresh_holdout.py FRESH_HOLDOUT2 results/FRESH_HOLDOUT_groups.csv
#   -> 이전 시험지의 사용자는 전부 빼서, 두 시험지가 사람 단위로 겹치지 않게 한다.
NAME = sys.argv[1] if len(sys.argv) > 1 else "FRESH_HOLDOUT"
EXCLUDE = sys.argv[2:]

r, groups = load_movielens()
r = r.drop(columns=["rating_raw", "rating_norm"])          # 별점은 아예 지워 두고 시작 (봉인)
trva = r[r.split.isin(["train", "validation"])]
te = r[r.split == "test"]
eval_users = set(groups[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())
hist = trva.groupby("user_id").size()
ok_users = set(hist[hist >= MIN_HISTORY].index) - eval_users
for path in EXCLUDE:
    prev = pd.read_csv(path)
    ok_users -= set(prev[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())
te = te[te.user_id.isin(ok_users) & te.movie_id.isin(trva.movie_id)]
sets = {u: set(d.movie_id) for u, d in te.groupby("user_id") if len(d) >= MIN_COMMON}
# (참고) 1·2번 시험지와 같은 탐색 방식

users = sorted(sets, key=lambda u: -len(sets[u]))
found = set()
for a in users:
    partners = [b for b in users if b != a and len(sets[a] & sets[b]) >= MIN_COMMON][:40]
    for trio in itertools.combinations(partners, 3):
        if len(sets[a] & sets[trio[0]] & sets[trio[1]] & sets[trio[2]]) >= MIN_COMMON:
            found.add(tuple(sorted((a, *trio))))
found = sorted(found)
pick = np.random.default_rng(SEED + 99 + 7 * len(EXCLUDE)).permutation(len(found))[:MAX_GROUPS]
rows = []
for gid, i in enumerate(sorted(pick), 1):
    g = found[i]
    common = sorted(set.intersection(*[sets[u] for u in g]))
    rows.append(dict(group_id=gid, user_id_1=g[0], user_id_2=g[1], user_id_3=g[2], user_id_4=g[3],
                     validation_common_movies="", test_common_movies=len(common), validation_movie_ids="",
                     test_movie_ids="|".join(map(str, common))))
out = pd.DataFrame(rows)
out.to_csv(RESULTS / f"{NAME}_groups.csv", index=False)
used = set(out[[f"user_id_{i}" for i in range(1, 5)]].values.ravel())
save_json(dict(n_groups=len(out), n_possible=len(found), n_users=len(used),
               avg_candidates=float(out.test_common_movies.mean()),
               note="봉인됨: 각 버전 최종 모델을 한 번씩만 채점. 이 파일을 만들 때 별점은 읽지 않음.",
               excluded_from=EXCLUDE, min_common=MIN_COMMON),
          f"{NAME}_info.json")
print(f"새 시험지: 가능한 그룹 {len(found)}개 중 {len(out)}개 봉인 (사용자 {len(used)}명, 후보 평균 "
      f"{out.test_common_movies.mean():.1f}편)")
