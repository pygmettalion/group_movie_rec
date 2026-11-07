# 4인 그룹 영화 추천: Strong Dislike 를 줄이는 Risk-Aware Group Recommender

> English version: `README.md`

4명이 같이 볼 영화 1편(Top-1)을 추천한다. 평균만 최대화하면 한 명이 소외될 수 있다.
이 모델은 **평균 만족도를 거의 유지하면서, 4명 중 한 명이라도 강하게 싫어하는(Strong Dislike, 강한 불호) 일을 줄이는 것**이 목표다.

## 최종 모델 = v1 (`results_v1_frozen/`, 고정본)
```
사용자 4명
→ Stage 1 Hybrid 로 후보 영화별 개인 선호 예측
→ 각 구성원의 Strong Dislike 확률 p_i 예측 (개인 단위 HGB 분류기)
→ Group Misery Risk = 1 − (1−p1)(1−p2)(1−p3)(1−p4)
→ Final Score = Expected Group Satisfaction(4명 예측 평균) − λ × Group Risk,  λ = 0.75
→ 후보 중 Final Score 1등 영화 추천
```
- **Stage 1 (Hybrid):** ALS-MF(latent k=64, reg=0.02, epochs=8) + 장르 Content 를 사용자 과거 평점 수에 따라 결합(H2_bucket):
  50개 미만 0.65×MF + 0.35×Content, 50개 이상 MF 1.0. **모든 값은 Validation 탐색으로 선택**(`01`, `01b`, 기록 `results_v1_frozen/stage1*.csv`).
  MF 의 latent vector 는 Group Risk 에 직접 넣지 않는다. Stage 2 에는 Stage 1 의 최종 예측 점수 하나만 전달한다.
- **Strong Dislike 기준:** 개인별 과거 평점의 하위 5% (별점 동점 고려). 후보 5·10·15·20·25% 를 Validation 에서 비교해 선택.
  기준선은 **예측 대상 기간 이전의 평점만으로** 계산한다(target rating 미사용, leakage 방지).
- **Stage 2 (member-level risk):** 사용자-영화마다 HistGradientBoosting 으로 개인 Strong Dislike 확률 p_i 를 예측한 뒤 위 식으로 결합.
  4명의 feature 를 한 번에 넣는 direct group classifier 도 Validation 에서 비교했지만 성능이 낮아(log loss 0.434 vs 0.423, AUC 0.819 vs 0.830) 채택하지 않았다.
  피처: 예측 선호, 개인 기준선, 기준선과의 거리, 평소 Strong Dislike 비율, 영화의 과거 평점 통계.
- **λ = 0.75:** Validation 그룹(공식 49 + 교차 적합 연습 600)에서 평균 만족도·최저 만족도·(1−Strong Dislike 비율)의 평균(balanced score)이 최대인 값.
  **Test / External 결과를 본 뒤 λ 를 포함한 어떤 값도 바꾸지 않았다.**

## 최종 평가 방법
1. 공식 Test(49그룹)와 External(MovieTweetings 44그룹)을 고정 모델로 한 번 채점.
2. 이후 개선 실험(v2~v4)을 했기 때문에, 공정한 비교를 위해 **새 시험 그룹(fresh holdout)을 별도로 봉인**했다(`06_make_fresh_holdout.py`).
   봉인할 때는 "누가 어떤 영화를 평가했는지"(평가 가능성)만 보고, **실제 rating 값은 읽지 않았다**(코드에서 rating 열을 먼저 삭제).
3. 봉인된 같은 후보 영화 집합에 v1 과 비교 방법들을 적용해 각자 1편을 고르게 한 뒤, **실제 held-out rating 으로 채점**했다.
   모델의 예측값을 정답 대신 쓴 평가는 없다.

## 핵심 결과 (v1)
| 시험 | 결과 |
|---|---|
| 새 시험지 1·2번 합계 1,000그룹 (후보 5편 이상) | Strong Dislike: Average 6.7% → **5.4%** (95% 구간 −2.2 ~ −0.4%p). 평균 만족도 차이 −0.002(구간이 0 포함) |
| 기존 방식 10가지와 비교 (`07_why_ours.py`) | Least Misery 와 위험 수준은 유사(5.3% vs 5.4%). Least Misery 는 Average 대비 평균 만족도가 유의하게 낮았고(−0.004), Risk-Aware 는 유의한 손실이 없었다 |
| 후보 수별 분석 (`09_by_candidates.py`) | 후보 10편 이상 그룹에서 Average 대비 Strong Dislike **약 36% 감소**(6.8% → 4.4%, 206그룹). 후보 3편에서는 1% |
| 공식 Internal Test 49그룹 | Strong Dislike 18.4% → 16.3% |
| External: MovieTweetings 44그룹 | 파이프라인이 외부 데이터에서 작동했고 같은 경향을 확인(Strong Dislike 2.3%). 그룹 수가 적어 일반화는 제한적 |
| 추가 검증: Beliefs 25,360건 | 실제 시청 후 평점이 아니라 **관람 전 예상 별점(userPredictRating)**에 대한 미시청 영화 사전 선호 예측. 모든 경우 사용자 평균보다 오차가 작음 |

**결론:** Risk-Aware 는 Average 대비 평균 만족도를 거의 유지하면서 Strong Dislike 발생을 줄였다.
Least Misery 와 비교하면 위험 수준은 유사했으며, 만족도와 위험 사이의 균형을 목표로 했다.

그래프: `figures/` (fig1 기존 방식 비교, fig8 후보 수별 효과 등)

## 실행 방법
```bash
pip install -r requirements.txt
./run_v1_all.sh                                    # 01 → 01b → 02 → 03 → 04 (최종 모델 전체, 2코어 약 40분)
RESULTS_DIR=results_v1_frozen python 05_demo.py 1  # 그룹 1에게 실제 추천 (제목은 data_extra/ml-1m/movies.dat 있으면 표시)
python 07_why_ours.py && python 08_make_plots.py && python 09_by_candidates.py   # 비교·그래프
```

## 파일 지도
**최종 모델 (1차)**
| 파일 | 하는 일 |
|---|---|
| `01_stage1_select.py` | 문서 기본 후보 비교: SGD-MF / Content / Hybrid(α 하나) |
| `01b_stage1_alternatives.py` | 추가 후보: ALS-MF, k=0, 평점 수별 Hybrid 등 → Stage 1 확정 |
| `02_stage2_risk.py` | Strong Dislike 기준(percentile) + 위험 분류기·피처 선택, λ 용 연습 그룹(교차 적합) |
| `03_select_lambda.py` | λ 선택 → `FINAL_FROZEN_CONFIG.json` 으로 고정 |
| `04_final_test.py` | Internal Test, MovieTweetings, Beliefs 를 한 번 채점 |
| `05_demo.py` | 실제 추천 데모 (4명이 안 본 영화 전체가 후보) |
| `06_make_fresh_holdout.py` | 별점을 보지 않고 새 시험지 봉인 |
| `07_why_ours.py` / `09_by_candidates.py` | 기존 방식 10가지 비교 / 후보 수별 효과 |
| `08_make_plots.py` | 그래프 |
| `src/common.py` | 데이터 읽기, 장르 벡터, 채점 함수 |
| `src/models.py` | MF(SGD), ALSMF, ContentModel, Stage1Predictor |
| `src/risk.py` | 불호 기준선, 위험 피처 |
| `src/group.py` | 최종 점수, 그룹 채점, 짝지은 부트스트랩 |
| `korean_comments/` | 같은 코드에 한국어 주석을 단 읽기용 사본 (팀원용, 실행 코드는 동일. 루트 코드는 영어 주석) |

**추가 실험 (최종 모델 아님, 결과는 각 폴더)**
| 파일 | 내용 | 결론 |
|---|---|---|
| `11~14_v2_*.py`, `src/features_v2.py`, `src/risk_v2.py`, `src/metadata.py` | 개봉 연도·평론가·감독 피처, 최근 평점 가중 | 새 영화 개인 예측 개선, 그룹 추천은 1차와 차이 없음 |
| `31~34_v3_*.py`, `src/knn.py`, `src/stage1_v3.py` | MF 앙상블, Item-kNN, 섞기 모델 | 개인 예측 개선(0.886→0.878), 그룹은 차이 없음 |
| `41~43_v4_*.py` | 점수 식을 팀 목표 기준으로 재선택 | 3번 시험지에선 나았지만 교차 확인·외부에선 아님 → 1차 유지 (아래 참고) |
| `51~52_ml10m_*.py` | MovieLens 10M 에 1차 설정 그대로 적용 | 개인 예측 잘 통함(0.78), 위험 모델은 0.5점 단위 데이터에 재학습 필요 |
| `53_mt_external_big.py` | 원본 MovieTweetings 로 외부 그룹 확대 시도 | 10그룹만 추가 가능(겹치는 영화 부족) |

### 참고: `results_v4/V4_FINAL_RESULTS.json` 의 `adopt_v4: true`
V4 의 `adopt_v4` 는 3번째 holdout(후보 평균 3편)만 기준으로 한 당시의 선택 결과이다. 이후 독립 holdout 교차 확인(`results_v4/v4_nested_estimate.csv`)과
external robustness(MovieTweetings, `43_v4_external.py`)를 추가로 확인한 결과 안정적인 우위가 없어, **최종 제출 모델은 v1 으로 유지하였다.** (해당 JSON 은 기록 보존을 위해 수정하지 않았다.)

## 전처리 코드
`preprocessing/prepare_team_data.py` (팀 전처리 담당 코드, 원본 그대로) — 원본 ZIP 들로부터 `data/` 의 입력 CSV 를 만든다.
```bash
python preprocessing/prepare_team_data.py --movielens ml-1m.zip --movietweetings MovieTweetings-master.zip \
    --beliefs ml_belief_2024_data_release_2.zip --links ml-latest-small.zip --output data
```
설명: `preprocessing/00_먼저읽기.txt`. 원본 데이터는 라이선스 때문에 포함하지 않는다(아래 링크에서 다운로드).

## 재현성 확인
`./run_v1_all.sh` 로 빈 폴더(`results_repro/`)에서 처음부터 다시 돌린 결과가 `results_v1_frozen/` 과 소수점까지 같다
(Stage 1 설정, 위험 모델, λ=0.75, Internal·MovieTweetings·Beliefs 결과 전부).

### 참고: 결과 폴더에 남아 있는 '극혐' 표현
개발 중에는 Strong Dislike 를 '극혐'(극혐률, 극혐선 등)이라고 불렀다. 제출 전에 코드 주석·출력 문구·문서의 이 표현을
`Strong Dislike`(불호 기준선 등)로 바꿨지만, **실행 코드와 숫자는 바꾸지 않았다.**
반면 `results/`, `results_v1_frozen/`, `results_repro/`, `results_v2/`~`results_v4/` 안의 로그(`log_*.txt`)와
설정 JSON(예: `FINAL_FROZEN_CONFIG.json` 의 `lambda_rule`)은 **당시 실행 기록이므로 수정하지 않았고**, 예전 표현이 그대로 남아 있다.
두 표현은 같은 뜻이다(극혐 = Strong Dislike, 극혐선 = 불호 기준선, 극혐률 = Strong Dislike 비율).
따라서 지금 코드를 다시 실행하면 설정 JSON 의 설명 문구만 새 표현으로 저장되고, 모든 숫자는 기존 기록과 같다.

## 과적합 방지
시간 순서 분할 · Validation 으로만 선택 · 고정 후 한 번 채점 · 별점 안 보고 봉인한 새 시험지 3개 · 사용자 단위 교차검증 · 정규화와 조기 종료.

## 데이터 (저장소에 포함하지 않음)
MovieLens 는 재배포가 금지되어 원본을 올리지 않는다. `data/` 에 팀 전처리 데이터(`preprocessing/prepare_team_data.py` 로 생성), `data_extra/` 에 아래 파일을 넣는다.
- MovieLens 1M·latest-small·10M: https://grouplens.org/datasets/movielens/
- MovieTweetings: https://github.com/sidooms/MovieTweetings
- IMDb `title.crew.tsv.gz`: https://datasets.imdbws.com/ · Rotten Tomatoes 영화 CSV (Kaggle)

라이브러리·함수 설명과 참고 문헌: **`LIBRARIES_AND_REFERENCES.md`**
