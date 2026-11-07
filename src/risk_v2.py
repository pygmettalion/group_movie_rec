"""[version 2] Additional features for the Strong Dislike risk model.

On top of the v1 features (individual predicted score, Strong Dislike threshold, distance, Strong Dislike frequency, the movie's past rating statistics)
  - meta : critic score (+ missing flag), release year, director effect      -> "is a movie panned by critics more likely to be strongly disliked by someone?"
  - gd   : the person's 'past Strong Dislike rate in this movie's genres'    -> "someone who gives 1 star to every horror movie"
All are computed only from history before the prediction time (hist).
"""
import numpy as np

from common import genre_vector
from features_v2 import risk_item_table, user_genre_dislike
from risk import INDIV_FEATURES, indiv_features

META_COLS = ["critic", "critic_missing", "year_z", "dir_effect"]
INDIV_V2_SETS = {
    "I_member": INDIV_FEATURES,
    "I_meta": INDIV_FEATURES + META_COLS,
    "I_gd": INDIV_FEATURES + ["u_gdis"],
    "I_meta_gd": INDIV_FEATURES + META_COLS + ["u_gdis"],
}


class RiskContext:
    def __init__(self, gd_hist, thr, genre_lookup, alpha, item_hist, kappa):
        """gd_hist: those users' ratings before the prediction time (including a genres column) -> Strong Dislike rate by genre
        item_hist: MovieLens past ratings used to compute movie information (director effect), thr: user -> Strong Dislike threshold,
        genre_lookup: movie -> genre string (MovieLens notation)."""
        self.items = risk_item_table(item_hist, kappa)
        self.gdis = user_genre_dislike(gd_hist, thr, alpha)
        self.genre_lookup = genre_lookup
        self._gcache = {}

    def gvec(self, m):
        if m not in self._gcache:
            v = (genre_vector(self.genre_lookup.get(m)) > 0).astype(float)
            self._gcache[m] = v / v.sum() if v.sum() > 0 else v
        return self._gcache[m]


def indiv_features_v2(pred, thr, share, movie_ids, user_ids, mstats, ctx, item_table=None):
    f = indiv_features(pred, thr, share, movie_ids, mstats)
    it = (item_table if item_table is not None else ctx.items).reindex(movie_ids)
    f["critic"] = it.critic.fillna(ctx.items.critic.mean()).to_numpy()
    f["critic_missing"] = it.critic_missing.fillna(1.0).to_numpy()
    f["year_z"] = it.year_z.fillna(0.0).to_numpy()
    f["dir_effect"] = it.dir_effect.fillna(0.0).to_numpy()
    f["u_gdis"] = [float(ctx.gvec(m) @ ctx.gdis[u]) if u in ctx.gdis else np.nan for u, m in zip(user_ids, movie_ids)]
    f["u_gdis"] = f.u_gdis.fillna(f.share)
    return f


def risk_proba_v2(risk, pred, thr, share, movie_ids, users, mstats, ctx, item_table=None):
    """pred/thr/share: (number of candidates, 4). users: the 4 users. individual probabilities p_i -> 1 - prod(1 - p_i)."""
    cols = []
    for j in range(pred.shape[1]):
        X = indiv_features_v2(pred[:, j], thr[:, j], share[:, j], movie_ids, [users[j]] * len(movie_ids),
                              mstats, ctx, item_table)
        cols.append(risk["model"].predict_proba(X[risk["features"]])[:, 1])
    return 1 - np.prod(1 - np.column_stack(cols), axis=1)
