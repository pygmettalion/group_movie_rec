# 사용 라이브러리·함수 설명과 참고 문헌

> 공지 요구사항: "수업에서 다루지 않은 모듈/클래스/라이브러리/함수/메서드(파라미터 포함)를 설명하고, 참고한 출처를 반드시 밝힐 것."
> 아래 표의 '어디서'는 이 저장소의 파일 이름이다. 설명은 이 프로젝트에서 **실제로 넘긴 파라미터 값** 기준이다.

## 1. NumPy (수치 계산)

| 함수 | 어디서 | 하는 일 / 파라미터 |
|---|---|---|
| `np.linalg.solve(A, b)` | `src/models.py` (ALSMF, ContentModel, fold_in) | 연립방정식 `A x = b` 의 정확한 해. 릿지 회귀의 닫힌 해 `(XᵀX + λI)⁻¹ Xᵀy` 를 역행렬 없이 계산할 때 사용 |
| `np.linalg.lstsq(A, b, rcond=None)` | `01b_*.py` (H3_stack), `31_*.py` | 최소제곱 해. `rcond=None` 은 작은 특이값을 자르는 기준을 NumPy 기본값으로 둔다는 뜻 |
| `np.random.default_rng(seed)` | 거의 모든 파일 | 시드를 고정한 난수 생성기 (재현성). `.normal(0, 0.05, shape)` 초기 좌표, `.permutation(n)` 섞기, `.choice(n, 4, replace=False)` 중복 없이 4명 뽑기, `.integers(0, 2, n)` 반반 나누기 |
| `np.add.at(a, idx, v)` | `src/models.py` (SGD-MF) | 같은 인덱스가 여러 번 나와도 값을 **누적**해서 더함 (일반 `a[idx] += v` 는 중복 시 한 번만 더해짐) |
| `np.einsum("ij,ij->i", P, Q)` | `src/models.py` | 행마다 내적 (사용자 좌표 · 영화 좌표) |
| `np.digitize(x, bins)` | `src/models.py` (H2_bucket) | 값이 어느 구간에 속하는지 번호 반환. `bins=[50, 200]` → 평점 수 <50 / 50~199 / 200+ |
| `np.searchsorted(a, v, side="left")` | `src/models.py`, `04_*.py` | 정렬된 배열에서 위치 찾기. Beliefs 에서 '응답 시각보다 앞선 평점'만 자를 때 사용 |
| `np.argsort(a, kind="stable")` | `src/models.py` | 정렬 순서. `stable` 은 같은 값의 원래 순서를 유지 |
| `np.argpartition(-s, k-1, axis=1)` / `np.take_along_axis` | `src/knn.py` | 전체 정렬 없이 상위 k개 이웃만 빠르게 고르기 / 고른 위치의 값 꺼내기 |
| `np.quantile`, `np.percentile` | `src/risk.py`, `src/group.py` | 분위수. 부트스트랩 95% 구간(2.5%, 97.5%) 계산 |

## 2. pandas (표 데이터)

| 함수 | 어디서 | 하는 일 / 파라미터 |
|---|---|---|
| `pd.read_csv(path, encoding="utf-8-sig")` | `src/common.py` | 팀 CSV 앞의 BOM 문자를 제거하며 읽기 |
| `pd.read_csv(..., sep="::", engine="python", names=[...], encoding="latin1")` | `src/metadata.py`, `51_*.py` | MovieLens 원본 `.dat` (구분자 `::`). 여러 글자 구분자는 `engine="python"` 필요, 옛 파일이라 `latin1` |
| `groupby(...).agg(["sum","count"])`, `.transform("mean")`, `.quantile(q)` | 여러 곳 | 사용자/영화별 합계·개수·평균·분위수 |
| `pd.factorize(values)` | `src/models.py` | 값 → 0부터 시작하는 정수 번호 (사용자·영화 인덱싱) |
| `merge`, `reindex`, `pivot`, `to_pickle` | 여러 곳 | 표 합치기 / 원하는 순서로 재배열 / 넓은 표로 펼치기 / 파이썬 객체로 저장 |

## 3. SciPy

| 함수 | 어디서 | 하는 일 / 파라미터 |
|---|---|---|
| `scipy.sparse.csr_matrix((data, (row, col)), shape=...)` | `src/knn.py` (3차 실험) | 대부분이 빈칸인 사용자×영화 행렬을 메모리 적게 저장. `R.T @ R` 로 영화끼리 유사도 계산 |

## 4. scikit-learn (머신러닝)

| 클래스/함수 | 어디서 | 하는 일 / 사용한 파라미터 |
|---|---|---|
| `HistGradientBoostingClassifier` | `02_*.py` (최종 위험 모델) | 결정트리 여러 개를 순서대로 쌓아 앞의 오차를 줄이는 Gradient Boosting. 연속값을 구간(히스토그램)으로 묶어 빠름. **최종:** `learning_rate=0.03`(한 트리가 반영되는 비율), `max_leaf_nodes=31`(트리 하나의 최대 잎 수 = 복잡도), `max_iter=200`(트리 개수), `l2_regularization=1.0`(잎 값 규제), `random_state=42` |
| `HistGradientBoostingRegressor` | `31_*.py` (3차 GBM 섞기 후보) | 위와 같은 원리의 회귀 버전. `min_samples_leaf=100`(잎 하나의 최소 샘플 수) |
| `RandomForestClassifier` | `02_*.py` (비교 후보) | 무작위로 만든 결정트리들의 투표. `n_estimators=150`(트리 수), `min_samples_leaf`(30/150), `max_features="sqrt"`(나눌 때 보는 피처 수 = √전체), `n_jobs=-1`(CPU 전부 사용) |
| `LogisticRegression(C, max_iter)` | `02_*.py` (비교 후보) | 로지스틱 회귀. `C` = 규제의 역수(작을수록 강한 규제), `max_iter` = 최적화 반복 상한 |
| `StandardScaler`, `make_pipeline` | `02_*.py` | 피처를 평균 0·표준편차 1로 맞춘 뒤 모델에 넣는 과정을 하나로 묶기 |
| `GroupKFold(n_splits=5).split(X, y, groups)` | `01b_*.py`, `11_*.py`, `31_*.py` | 같은 그룹(사용자)이 학습과 채점에 동시에 들어가지 않게 나누는 교차검증 |
| `log_loss`, `roc_auc_score`, `average_precision_score`, `brier_score_loss` | `02_*.py` | 확률 예측 채점: 확률이 얼마나 정확한가(log loss, Brier) / 위험한 것을 위에 올리는가(AUC, PR-AUC) |

## 5. Matplotlib (그래프)

| 함수 | 어디서 | 설명 |
|---|---|---|
| `plt.subplots`, `ax.bar`, `ax.plot`, `fig.savefig(dpi=180)` | `08_*.py`, `09_*.py` | 막대·선 그래프, 고해상도 PNG 저장 |
| `matplotlib.use("Agg")` | 같은 곳 | 화면 없이 파일로만 그리는 모드 |
| `font_manager.addfont(path)` | 같은 곳 | 한글 표시용 Noto Sans CJK 글꼴 등록 |

## 6. 파이썬 표준 라이브러리

| 모듈 | 어디서 | 설명 |
|---|---|---|
| `itertools.combinations(seq, 3)` | `02_*.py`, `06_*.py` 등 | 3명 조합을 중복 없이 나열 (그룹 찾기) |
| `collections.Counter`, `defaultdict` | `51_*.py`, `53_*.py`, `src/features_v2.py` | 개수 세기 / 기본값 있는 사전 |
| `pickle`, `json` | 여러 곳 | 학습된 모델 저장 / 설정·결과 저장 |
| `gzip`, `zipfile` | `src/metadata.py`, `51_*.py` | 압축 파일을 풀지 않고 바로 읽기 |
| `unicodedata.normalize("NFKD", s)`, `re` | `src/metadata.py` | 영화 제목의 악센트 제거·정규식으로 제목 정리 (Rotten Tomatoes 짝 맞추기) |

## 7. 직접 구현한 알고리즘 (라이브러리 아님)

| 이름 | 파일 | 한 줄 설명 |
|---|---|---|
| 편향 포함 MF (SGD, ALS) | `src/models.py` | 예측 = 전체평균 + 사용자편향 + 영화편향 + 사용자좌표·영화좌표. ALS 는 한쪽 고정 후 릿지 회귀로 번갈아 풂 (규제 = reg × 평점 수) |
| Fold-in | `src/models.py` `MF.fold_in` | 새 사용자(MovieTweetings, Beliefs)의 좌표를 영화 좌표 고정 상태로 릿지 회귀로 계산 |
| 장르 Content 모델 | `src/models.py` `ContentModel` | 사용자 평균(전체 평균 쪽으로 당김) + 사람별 장르 가중치(전체 공통 가중치 쪽으로 당김) |
| 동점 고려 불호 기준선 | `src/risk.py` `user_thresholds` | '그 점수 이하가 그 사람 평점의 p% 를 넘지 않는 가장 높은 점수' |
| 그룹 위험 | `src/group.py` | 1 − ∏(1 − pᵢ) |
| 짝지은 부트스트랩 | `src/group.py` `paired_bootstrap` | 같은 그룹에서 두 방법의 차이를 5,000번 다시 뽑아 95% 구간 계산 |
| Item-kNN | `src/knn.py` | 기준 점수 + 비슷한 영화 k개의 '취향 나머지' 가중평균 |

## 8. 참고 문헌과 데이터 출처

**데이터**
- F. M. Harper, J. A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4), 2015. doi:10.1145/2827872 (MovieLens 1M, 10M, latest-small) — https://grouplens.org/datasets/movielens/
- S. Dooms, T. De Pessemier, L. Martens. *MovieTweetings: a Movie Rating Dataset Collected From Twitter.* CrowdRec Workshop @ RecSys, 2013. doi:10.1145/2559858.2559862 — https://github.com/sidooms/MovieTweetings
- MovieLens Beliefs 데이터 (GroupLens, `ml_belief_2024_data_release_2`)
- IMDb Non-Commercial Datasets (`title.crew.tsv.gz`) — https://datasets.imdbws.com/
- Rotten Tomatoes 영화 평점 CSV (Kaggle 공개 데이터셋, 2차 실험)

**방법**
- Y. Koren, R. Bell, C. Volinsky. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer, 2009. (편향 포함 MF)
- Y. Zhou, D. Wilkinson, R. Schreiber, R. Pan. *Large-Scale Parallel Collaborative Filtering for the Netflix Prize.* AAIM 2008, LNCS 5034. doi:10.1007/978-3-540-68880-8_32 (ALS, 평점 수에 비례한 규제)
- B. Sarwar, G. Karypis, J. Konstan, J. Riedl. *Item-Based Collaborative Filtering Recommendation Algorithms.* WWW, 2001. (Item-kNN)
- J. Masthoff. *Group Recommender Systems: Combining Individual Models.* Recommender Systems Handbook, Springer, 2011. doi:10.1007/978-0-387-85820-3_21 (Average, Least Misery, Average Without Misery, Borda, Multiplicative, Most Pleasure)
- J. H. Friedman. *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 2001.
- B. Efron, R. Tibshirani. *An Introduction to the Bootstrap.* 1993.
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011.

**코드**
- 인터넷 코드를 복사해 쓰지 않았다. 위 논문들의 식을 NumPy 로 직접 구현했다.
