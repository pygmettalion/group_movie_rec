"""Group recommendation scoring and evaluation.

Final Score = Expected Group Satisfaction - lambda * Predicted Group Misery Risk
  - Expected Group Satisfaction: mean of the 4 predicted preference scores
  - Predicted Group Misery Risk: the Stage 2 classifier's 'probability that someone strongly dislikes it'

Evaluation (after the answers are revealed)
  - average satisfaction: mean of the real scores (0~1) the 4 people gave the recommended movie
  - minimum satisfaction: the score of the person who rated it lowest among the 4
  - Strong Dislike occurrence: did at least one of the 4 give a score at or below their own Strong Dislike threshold
"""
import numpy as np
import pandas as pd

from risk import group_features, indiv_features

LAMBDA_GRID = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def risk_proba(risk, pred, thr, share, movie_ids, mstats):
    """kind='group': the group classifier gives the probability directly / kind='indiv': combine the 4 individual probabilities p_i as 1-prod(1-p_i)."""
    if risk.get("kind", "group") == "group":
        X = group_features(pred, thr, share, movie_ids, mstats)
        return risk["model"].predict_proba(X[risk["features"]])[:, 1]
    pi = np.column_stack([
        risk["model"].predict_proba(indiv_features(pred[:, j], thr[:, j], share[:, j], movie_ids, mstats)
                                    [risk["features"]])[:, 1] for j in range(pred.shape[1])])
    return 1 - np.prod(1 - pi, axis=1)


def build_case(group_id, users, movies, pred, true, thr, share, risk, mstats):
    """Bundle the information of one group's candidate movies. pred/true: (number of candidates, 4)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    thr4 = np.tile(thr, (len(movies), 1))
    share4 = np.tile(share, (len(movies), 1))
    return dict(group_id=group_id, users=list(users), movies=list(movies), pred=pred, true=true, thr=np.asarray(thr),
                exp_sat=pred.mean(1), risk=risk_proba(risk, pred, thr4, share4, np.asarray(movies), mstats))


def outcome(case, i):
    t = case["true"][i]
    return dict(avg=float(t.mean()), min=float(t.min()), sd=float((t <= case["thr"]).any()))


def choose(case, method, lam=0.0):
    if method == "Average":
        s = case["exp_sat"]
    elif method == "LeastMisery":
        s = case["pred"].min(1) + 1e-9 * case["exp_sat"]
    elif method == "RiskAware":
        s = case["exp_sat"] - lam * case["risk"]
    elif method == "Oracle":
        s = case["true"].mean(1)
    else:
        raise ValueError(method)
    return int(np.argmax(s))


def evaluate(cases, method, lam=0.0):
    """Per-group results table. Random is the expected value of 'any candidate' (mean over all candidates)."""
    rows = []
    for c in cases:
        if method == "Random":
            outs = [outcome(c, i) for i in range(len(c["movies"]))]
            o = {k: float(np.mean([x[k] for x in outs])) for k in ("avg", "min", "sd")}
            pick = None
        else:
            pick = choose(c, method, lam)
            o = outcome(c, pick)
        rows.append(dict(group_id=c["group_id"], method=method, lam=lam,
                         pick=None if pick is None else c["movies"][pick], **o))
    return pd.DataFrame(rows)


def summarize(df):
    s = dict(avg_satisfaction=df.avg.mean(), min_satisfaction=df["min"].mean(), strong_dislike_rate=df.sd.mean())
    s["balanced_score"] = (s["avg_satisfaction"] + s["min_satisfaction"] + (1 - s["strong_dislike_rate"])) / 3
    return s


def paired_bootstrap(a, b, n=5000, seed=42):
    """Difference method a - method b on the same groups. Groups are resampled to compute the 95% interval."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in ("avg", "min", "sd"):
        d = a[k].to_numpy() - b[k].to_numpy()
        boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
        out[k] = dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))
    return out
