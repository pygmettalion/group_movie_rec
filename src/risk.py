"""Stage 2 tools: the Strong Dislike threshold and Group Risk (risk of a 'Strong Dislike' within the group) features.

Strong Dislike (a relative threshold per person)
  - Rating habits differ. For a stingy rater 3 stars may mean 'meh', for a generous one 3 stars may mean 'awful'.
  - So Strong Dislike is defined as "a score at or below the bottom p% line (threshold) of the scores the person gave in the past".
  - The threshold is always computed only from 'history before the prediction period' (no leakage of the answer ratings).

Group Misery Risk
  - A classifier predicts "the probability that at least one of the 4 people strongly dislikes it".
  - The classifier inputs are only Stage 1's final individual preference score (+ per-person threshold and the movie's past rating statistics).
"""
import numpy as np
import pandas as pd

PERCENTILES = [5, 10, 15, 20, 25]


def user_thresholds(hist, p):
    """hist: ratings before the prediction time. Returns: DataFrame(user_id -> thr, share)

    Star ratings are discrete like 1,2,3,4,5, so many scores are equal (ties).
    With the usual percentile formula (<= q_p), 'bottom 5%' actually catches 14% of a person's ratings (checked on the data).
    So the Strong Dislike threshold is 'the highest score such that ratings at or below it do not exceed p% of that person's ratings'.
    (For a person whose single lowest score already exceeds p%, that lowest score is used as the threshold.)
    share = share of past ratings at or below the threshold (how often the person gives a 'Strong Dislike').
    """
    cnt = hist.groupby(["user_id", "rating_norm"]).size().rename("n").reset_index()
    cnt["cum"] = cnt.groupby("user_id").n.cumsum() / cnt.groupby("user_id").n.transform("sum")
    ok = cnt[cnt.cum <= p / 100.0 + 1e-12].groupby("user_id").rating_norm.max()
    lowest = cnt.groupby("user_id").rating_norm.min()
    thr = ok.reindex(lowest.index).fillna(lowest)
    share = cnt.assign(thr=cnt.user_id.map(thr)).query("rating_norm <= thr").groupby("user_id").cum.max()
    return pd.DataFrame({"thr": thr, "share": share.reindex(thr.index)})


def movie_stats(hist):
    """Per-movie past rating statistics (an 'observed data summary', not a model-internal representation). Low-rating share = the bottom 25% range."""
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
    """pred/thr/share: arrays of shape (number of group-movie pairs, 4). margin = predicted score - that person's Strong Dislike threshold (negative means risky)."""
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


# for the individual-level risk model (additional candidate structure): compute a 'Strong Dislike probability' for each person-movie pair.
INDIV_FEATURES = ["pred", "thr", "margin", "share", "m_logcount", "m_mean", "m_std", "m_lowfrac"]


def indiv_features(pred, thr, share, movie_ids, mstats):
    pred, thr, share = (np.asarray(a, float) for a in (pred, thr, share))
    f = pd.DataFrame({"pred": pred, "thr": thr, "margin": pred - thr, "share": share})
    ms = mstats.reindex(movie_ids)
    global_fill = mstats.mean()
    for c in ["m_logcount", "m_mean", "m_std", "m_lowfrac"]:
        f[c] = ms[c].fillna(0.0 if c == "m_logcount" else global_fill[c]).to_numpy()
    return f
