# Libraries, Functions, and References

> 한국어 버전: `LIBRARIES_AND_REFERENCES_ko.md`
>
> Assignment requirement: "Include explanations for all modules/classes/libraries/functions/methods (along with the parameters) used that were not taught or used in classes, and cite every source."
> The "Where" column is a file name in this repository. Explanations are based on **the parameter values actually passed** in this project.

## 1. NumPy (numerical computing)

| Function | Where | What it does / parameters |
|---|---|---|
| `np.linalg.solve(A, b)` | `src/models.py` (ALSMF, ContentModel, fold_in) | Exact solution of the linear system `A x = b`. Used to compute the closed-form ridge solution `(XᵀX + λI)⁻¹ Xᵀy` without forming an inverse |
| `np.linalg.lstsq(A, b, rcond=None)` | `01b_*.py` (H3_stack), `31_*.py` | Least-squares solution. `rcond=None` uses NumPy's default cutoff for small singular values |
| `np.random.default_rng(seed)` | almost every file | Random generator with a fixed seed (reproducibility). `.normal(0, 0.05, shape)` initial factors, `.permutation(n)` shuffling, `.choice(n, 4, replace=False)` pick 4 without replacement, `.integers(0, 2, n)` split in half |
| `np.add.at(a, idx, v)` | `src/models.py` (SGD-MF) | **Accumulates** values even when the same index appears several times (plain `a[idx] += v` adds only once for duplicates) |
| `np.einsum("ij,ij->i", P, Q)` | `src/models.py` | Row-wise dot product (user factors · movie factors) |
| `np.digitize(x, bins)` | `src/models.py` (H2_bucket) | Returns the bin index of each value. `bins=[50, 200]` → rating count <50 / 50–199 / 200+ |
| `np.searchsorted(a, v, side="left")` | `src/models.py`, `04_*.py` | Position in a sorted array. Used for Beliefs to keep only ratings earlier than the response time |
| `np.argsort(a, kind="stable")` | `src/models.py` | Sort order. `stable` keeps the original order of equal values |
| `np.argpartition(-s, k-1, axis=1)` / `np.take_along_axis` | `src/knn.py` | Quickly select only the top-k neighbors without a full sort / take the values at the selected positions |
| `np.percentile(boots, 2.5 / 97.5)` | `src/group.py` | Percentiles. Computes the bootstrap 95% interval (2.5%, 97.5%) |

## 2. pandas (tabular data)

| Function | Where | What it does / parameters |
|---|---|---|
| `pd.read_csv(path, encoding="utf-8-sig")` | `src/common.py` | Reads the team CSVs while removing the leading BOM character |
| `pd.read_csv(..., sep="::", engine="python", names=[...], encoding="latin1")` | `src/metadata.py`, `51_*.py` | Original MovieLens `.dat` files (separator `::`). A multi-character separator needs `engine="python"`; old files use `latin1` |
| `groupby(...).agg(["sum","count"])`, `.transform("mean")`, `.quantile(q)` | many places | Per-user / per-movie sums, counts, means, quantiles |
| `pd.factorize(values)` | `src/models.py` | Values → integer codes starting at 0 (user and movie indexing) |
| `merge`, `reindex`, `pivot`, `to_pickle` | many places | Join tables / reorder / reshape to wide format / save Python objects |

## 3. SciPy

| Function | Where | What it does / parameters |
|---|---|---|
| `scipy.sparse.csr_matrix((data, (row, col)), shape=...)` | `src/knn.py` (v3 experiment) | Stores the mostly empty user×movie matrix with little memory. `R.T @ R` computes movie–movie similarity |

## 4. scikit-learn (machine learning)

| Class / function | Where | What it does / parameters used |
|---|---|---|
| `HistGradientBoostingClassifier` | `02_*.py` (final risk model) | Gradient boosting: decision trees added one after another, each reducing the previous errors. Continuous values are binned (histograms), so it is fast. **Final:** `learning_rate=0.03` (how much each tree contributes), `max_leaf_nodes=31` (max leaves per tree = complexity), `max_iter=200` (number of trees), `l2_regularization=1.0` (leaf-value regularization), `random_state=42` |
| `HistGradientBoostingRegressor` | `31_*.py` (v3 GBM blender candidate) | Regression version of the above. `min_samples_leaf=100` (minimum samples per leaf) |
| `RandomForestClassifier` | `02_*.py` (comparison candidate) | Vote of randomly built decision trees. `n_estimators=150` (number of trees), `min_samples_leaf` (30/150), `max_features="sqrt"` (features considered per split = √total), `n_jobs=-1` (use all CPUs) |
| `LogisticRegression(C, max_iter)` | `02_*.py` (comparison candidate) | Logistic regression. `C` = inverse of regularization (smaller = stronger), `max_iter` = optimizer iteration limit |
| `StandardScaler`, `make_pipeline` | `02_*.py` | Bundle "scale features to mean 0 / std 1, then fit the model" into one step |
| `GroupKFold(n_splits=5).split(X, y, groups)` | `01b_*.py`, `11_*.py`, `31_*.py` | Cross-validation in which the same group (user) never appears in both training and scoring |
| `log_loss`, `roc_auc_score`, `average_precision_score`, `brier_score_loss` | `02_*.py` | Scoring probability predictions: how accurate the probabilities are (log loss, Brier) / how well risky items are ranked on top (AUC, PR-AUC) |

## 5. Matplotlib (figures)

| Function | Where | Description |
|---|---|---|
| `plt.subplots`, `ax.bar`, `ax.plot`, `fig.savefig(dpi=180)` | `08_*.py`, `09_*.py` | Bar and line charts, saved as high-resolution PNG |
| `matplotlib.use("Agg")` | same | Draw to files only, without a screen |
| `font_manager.addfont(path)` | same | Registers the Noto Sans CJK font to display Korean labels |

## 6. Python standard library

| Module | Where | Description |
|---|---|---|
| `itertools.combinations(seq, 3)` | `02_*.py`, `06_*.py`, etc. | Enumerates 3-person combinations without repetition (group search) |
| `collections.Counter`, `defaultdict` | `51_*.py`, `53_*.py`, `src/features_v2.py` | Counting / dictionary with default values |
| `pickle`, `json` | many places | Save trained models / save settings and results |
| `gzip`, `zipfile` | `src/metadata.py`, `51_*.py` | Read compressed files directly without extracting |
| `unicodedata.normalize("NFKD", s)`, `re` | `src/metadata.py` | Remove accents from movie titles and clean titles with regular expressions (matching with Rotten Tomatoes) |

## 7. Algorithms implemented by hand (not libraries)

| Name | File | One-line description |
|---|---|---|
| Biased MF (SGD, ALS) | `src/models.py` | prediction = global mean + user bias + movie bias + user factors · movie factors. ALS fixes one side and solves the other by ridge regression, alternately (regularization = reg × number of ratings) |
| Fold-in | `src/models.py` `MF.fold_in` | Computes factors of a new user (MovieTweetings, Beliefs) by ridge regression with the movie factors fixed |
| Genre Content model | `src/models.py` `ContentModel` | User mean (shrunk toward the global mean) + per-person genre weights (shrunk toward the shared global weights) |
| Tie-aware Strong Dislike threshold | `src/risk.py` `user_thresholds` | 'The highest score such that ratings at or below it do not exceed p% of that person's ratings' |
| Group risk | `src/group.py` | 1 − ∏(1 − pᵢ) |
| Paired bootstrap | `src/group.py` `paired_bootstrap` | Resamples the per-group difference between two methods 5,000 times to compute a 95% interval |
| Item-kNN | `src/knn.py` | Baseline score + similarity-weighted mean of the 'taste residuals' of the k most similar movies |

## 8. References and data sources

**Data**
- F. M. Harper, J. A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4), 2015. doi:10.1145/2827872 (MovieLens 1M, 10M, latest-small) — https://grouplens.org/datasets/movielens/
- S. Dooms, T. De Pessemier, L. Martens. *MovieTweetings: a Movie Rating Dataset Collected From Twitter.* CrowdRec Workshop @ RecSys, 2013. doi:10.1145/2559858.2559862 — https://github.com/sidooms/MovieTweetings
- MovieLens Beliefs data (GroupLens, `ml_belief_2024_data_release_2`)
- IMDb Non-Commercial Datasets (`title.crew.tsv.gz`) — https://datasets.imdbws.com/
- Rotten Tomatoes movie ratings CSV (public Kaggle dataset, v2 experiment)

**Methods**
- Y. Koren, R. Bell, C. Volinsky. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer, 2009. (biased MF)
- Y. Zhou, D. Wilkinson, R. Schreiber, R. Pan. *Large-Scale Parallel Collaborative Filtering for the Netflix Prize.* AAIM 2008, LNCS 5034. doi:10.1007/978-3-540-68880-8_32 (ALS, regularization proportional to the number of ratings)
- B. Sarwar, G. Karypis, J. Konstan, J. Riedl. *Item-Based Collaborative Filtering Recommendation Algorithms.* WWW, 2001. (Item-kNN)
- J. Masthoff. *Group Recommender Systems: Combining Individual Models.* Recommender Systems Handbook, Springer, 2011. doi:10.1007/978-0-387-85820-3_21 (Average, Least Misery, Average Without Misery, Borda, Multiplicative, Most Pleasure)
- J. H. Friedman. *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 2001.
- B. Efron, R. Tibshirani. *An Introduction to the Bootstrap.* 1993.
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011.

**Code**
- No code was copied from the internet. The formulas from the papers above were implemented directly with NumPy.
