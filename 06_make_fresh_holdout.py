"""Build a 'fresh holdout' (new test set) for version 2 — only seal it, never look at the answers.

Why?
  The Internal/External Test scores of version 1 have already been seen. If the model is changed afterwards, that test set no longer gives a fair comparison.
  So new groups that 'nobody has ever scored' are built from the Test period and sealed.
  - the 36 users of the 49 official groups are excluded
  - same conditions as the official groups: 4 people with >= 5 common movies in the Test period, each with >= 10 Train+Validation ratings
  - this script only looks at 'who watched which movie' and never opens the ratings
  - version 1 and version 2 are each scored 'exactly once' on this test set and compared
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import RESULTS, SEED, load_movielens, save_json  # noqa: E402

MIN_COMMON = int(os.environ.get("MIN_COMMON", 5))      # test set 3 uses 3 because the remaining users have little history
MIN_HISTORY = 10
MAX_GROUPS = int(os.environ.get("MAX_GROUPS", 500))
# usage: python 06_make_fresh_holdout.py [name] [previous_test_set_to_exclude.csv ...]
#   for v3: python 06_make_fresh_holdout.py FRESH_HOLDOUT2 results/FRESH_HOLDOUT_groups.csv
#   -> all users of the previous test sets are removed, so the test sets never share a person
NAME = sys.argv[1] if len(sys.argv) > 1 else "FRESH_HOLDOUT"
EXCLUDE = sys.argv[2:]

r, groups = load_movielens()
r = r.drop(columns=["rating_raw", "rating_norm"])          # start with the ratings deleted (sealed)
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
# (reference) the same search procedure as test sets 1 and 2

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
