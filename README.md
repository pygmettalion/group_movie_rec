# Group Movie Recommendation for 4 People: a Risk-Aware Group Recommender that Reduces Strong Dislike

> 한국어 버전: `README_ko.md`

The system recommends one movie (Top-1) for a group of 4 people to watch together. Maximizing only the average can leave one member out.
The goal of this model is to **keep the average satisfaction almost unchanged while reducing the chance that at least one of the 4 members strongly dislikes the movie (Strong Dislike)**.

## Final model = v1 (`results_v1_frozen/`, frozen)
```
4 users
→ Stage 1 Hybrid predicts each member's preference for every candidate movie
→ Predict each member's Strong Dislike probability p_i (member-level HGB classifier)
→ Group Misery Risk = 1 − (1−p1)(1−p2)(1−p3)(1−p4)
→ Final Score = Expected Group Satisfaction (mean of the 4 predictions) − λ × Group Risk,  λ = 0.75
→ Recommend the candidate with the highest Final Score
```
- **Stage 1 (Hybrid):** ALS-MF (latent k=64, reg=0.02, epochs=8) + genre-based Content model, combined by the user's number of past ratings (H2_bucket):
  fewer than 50 ratings → 0.65×MF + 0.35×Content, 50 or more → MF 1.0. **All values were selected by validation search** (`01`, `01b`, logs in `results_v1_frozen/stage1*.csv`).
  The MF latent vectors are not fed into the Group Risk model. Stage 2 receives only the single final prediction score from Stage 1.
- **Strong Dislike threshold:** the bottom 5% of each user's own past ratings (tie-aware). Candidates 5·10·15·20·25% were compared on validation.
  The threshold is computed **only from ratings before the prediction period** (target ratings are never used; no leakage).
- **Stage 2 (member-level risk):** for each user–movie pair, HistGradientBoosting predicts the member's Strong Dislike probability p_i, and the 4 probabilities are combined with the formula above.
  A direct group classifier that takes the features of all 4 members at once was also compared on validation, but it performed worse (log loss 0.434 vs 0.423, AUC 0.819 vs 0.830), so it was not adopted.
  Features: predicted preference, personal threshold, distance to the threshold, the user's usual Strong Dislike rate, and the movie's past rating statistics.
- **λ = 0.75:** the value that maximizes the balanced score — the mean of average satisfaction, minimum satisfaction, and (1 − Strong Dislike rate) — on validation groups (49 official + 600 cross-fitted practice groups).
  **No value, including λ, was changed after seeing the Test / External results.**

## Final evaluation procedure
1. The official Test (49 groups) and External test (MovieTweetings, 44 groups) were scored once with the frozen model.
2. Because improvement experiments (v2–v4) were run afterwards, **new test groups (fresh holdouts) were sealed separately** for a fair comparison (`06_make_fresh_holdout.py`).
   When sealing, only "who rated which movie" (availability for evaluation) was used; **the actual rating values were not read** (the code drops the rating column first).
3. v1 and all comparison methods were applied to the same sealed candidate sets, each picking one movie, which was then **scored with the actual held-out ratings**.
   Model predictions were never used in place of ground truth.

## Key results (v1)
| Test | Result |
|---|---|
| Fresh holdouts 1+2, 1,000 groups (≥5 candidates) | Strong Dislike: Average 6.7% → **5.4%** (95% CI −2.2 to −0.4 pp). Average-satisfaction difference −0.002 (CI includes 0) |
| Comparison with 10 existing methods (`07_why_ours.py`) | Risk level similar to Least Misery (5.3% vs 5.4%). Least Misery had significantly lower average satisfaction than Average (−0.004); Risk-Aware had no significant loss |
| Analysis by number of candidates (`09_by_candidates.py`) | In groups with 10+ candidates, Strong Dislike **reduced by about 36%** vs Average (6.8% → 4.4%, 206 groups). 1% with 3 candidates |
| Official Internal Test, 49 groups | Strong Dislike 18.4% → 16.3% |
| External: MovieTweetings, 44 groups | The pipeline worked on external data and showed a similar trend (Strong Dislike 2.3%). Few groups, so generalization is limited |
| Additional check: Beliefs, 25,360 responses | Not post-viewing ratings but **pre-viewing expected ratings (userPredictRating)**: preference prediction for unseen movies. Error lower than the user-mean baseline in all cases |

**Conclusion:** Compared with Average, Risk-Aware reduced Strong Dislike while keeping average satisfaction almost unchanged.
Compared with Least Misery, the risk level was similar; the aim was a balance between satisfaction and risk.

Figures: `figures/` (fig1 comparison with existing methods, fig8 effect by number of candidates, etc.)

## How to run
```bash
pip install -r requirements.txt
./run_v1_all.sh                                    # 01 → 01b → 02 → 03 → 04 (whole final model, ~40 min on 2 cores)
RESULTS_DIR=results_v1_frozen python 05_demo.py 1  # real recommendation for group 1 (titles shown if data_extra/ml-1m/movies.dat exists)
python 07_why_ours.py && python 08_make_plots.py && python 09_by_candidates.py   # comparisons and figures
```

## File map
**Final model (v1)**
| File | What it does |
|---|---|
| `01_stage1_select.py` | Compare the baseline candidates from the design doc: SGD-MF / Content / Hybrid (single α) |
| `01b_stage1_alternatives.py` | Additional candidates: ALS-MF, k=0, Hybrid by rating count, etc. → fixes Stage 1 |
| `02_stage2_risk.py` | Strong Dislike threshold (percentile) + risk classifier and feature selection, practice groups for λ (cross-fitted) |
| `03_select_lambda.py` | Select λ → freeze into `FINAL_FROZEN_CONFIG.json` |
| `04_final_test.py` | Score Internal Test, MovieTweetings, and Beliefs once |
| `05_demo.py` | Real recommendation demo (all movies none of the 4 has seen are candidates) |
| `06_make_fresh_holdout.py` | Seal new test groups without looking at ratings |
| `07_why_ours.py` / `09_by_candidates.py` | Comparison with 10 existing methods / effect by number of candidates |
| `08_make_plots.py` | Figures |
| `src/common.py` | Data loading, genre vectors, scoring functions |
| `src/models.py` | MF (SGD), ALSMF, ContentModel, Stage1Predictor |
| `src/risk.py` | Strong Dislike threshold, risk features |
| `src/group.py` | Final score, group scoring, paired bootstrap |

**Additional experiments (not the final model; results in each folder)**
| Files | Content | Conclusion |
|---|---|---|
| `11~14_v2_*.py`, `src/features_v2.py`, `src/risk_v2.py`, `src/metadata.py` | Release year, critic score, director features; recency-weighted ratings | Better individual prediction for new movies; no difference from v1 in group recommendation |
| `31~34_v3_*.py`, `src/knn.py`, `src/stage1_v3.py` | MF ensemble, Item-kNN, blending models | Better individual prediction (0.886→0.878); no difference in group recommendation |
| `41~43_v4_*.py` | Re-selected the scoring formula by the team goal | Better on fresh holdout 3, but not in cross-checks or on external data → v1 kept (see below) |
| `51~52_ml10m_*.py` | Applied the v1 settings as-is to MovieLens 10M | Individual prediction transfers well (0.78); the risk model needs retraining for half-star data |
| `53_mt_external_big.py` | Tried to enlarge the external groups using raw MovieTweetings | Only 10 more groups possible (not enough overlapping movies) |

### Note: `adopt_v4: true` in `results_v4/V4_FINAL_RESULTS.json`
The `adopt_v4` flag is the decision made at that time based only on the 3rd holdout (3 candidates per group on average). After additional independent-holdout cross-checks (`results_v4/v4_nested_estimate.csv`)
and an external robustness check (MovieTweetings, `43_v4_external.py`) showed no stable advantage, **the final submitted model was kept as v1.** (The JSON file was left unmodified to preserve the record.)

## Preprocessing code
`preprocessing/prepare_team_data.py` (the team's preprocessing code, unchanged) — builds the input CSVs in `data/` from the original ZIP files.
```bash
python preprocessing/prepare_team_data.py --movielens ml-1m.zip --movietweetings MovieTweetings-master.zip \
    --beliefs ml_belief_2024_data_release_2.zip --links ml-latest-small.zip --output data
```
Description: `preprocessing/00_먼저읽기.txt` ("read me first"). The original data is not included because of its license (download links below).

## Reproducibility check
Running `./run_v1_all.sh` from an empty folder (`results_repro/`) reproduces `results_v1_frozen/` exactly, down to the decimals
(Stage 1 settings, risk model, λ=0.75, and all Internal, MovieTweetings, and Beliefs results).

### Note: the Korean word '극혐' left in the result folders
During development, Strong Dislike was called '극혐' (Korean slang for "strongly dislike"; also 극혐률, 극혐선, etc.). Before submission, this wording in code comments, printed messages, and documents was changed to
`Strong Dislike` (Strong Dislike threshold, etc.), but **no executable code or number was changed.**
On the other hand, the logs (`log_*.txt`) and config JSON files (e.g., `lambda_rule` in `FINAL_FROZEN_CONFIG.json`) in `results/`, `results_v1_frozen/`, `results_repro/`, and `results_v2/`–`results_v4/`
**are records of past runs, so they were not modified**, and the old wording remains there.
The two terms mean the same thing (극혐 = Strong Dislike, 극혐선 = Strong Dislike threshold, 극혐률 = Strong Dislike rate).
If the code is run again, only the description text in the config JSON is saved with the new wording; all numbers are identical to the existing records.

## Preventing overfitting
Time-ordered split · selection on validation only · scored once after freezing · 3 fresh holdouts sealed without looking at ratings · user-grouped cross-validation · regularization and early stopping.

## Data (not included in this repository)
MovieLens may not be redistributed, so the original data is not uploaded. Put the team's preprocessed data in `data/` (generated by `preprocessing/prepare_team_data.py`) and the files below in `data_extra/`.
- MovieLens 1M · latest-small · 10M: https://grouplens.org/datasets/movielens/
- MovieTweetings: https://github.com/sidooms/MovieTweetings
- IMDb `title.crew.tsv.gz`: https://datasets.imdbws.com/ · Rotten Tomatoes movie CSV (Kaggle)

Library / function explanations and references: **`LIBRARIES_AND_REFERENCES.md`**
