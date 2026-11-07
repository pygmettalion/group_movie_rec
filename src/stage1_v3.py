"""[version 3] Stage 1: combine MF (mean of several training runs) + Content (genre, year, critic, director) + Item-kNN with a 'blending model'.

The blender type is chosen by 31_v3_stage1.py with Validation cross-validation:
  H2      : the same 'ratio per rating-count bucket' as v1/v2 (MF + Content)
  LIN3    : a + b*MF + c*Content + d*kNN
  LIN3_B  : the formula above, separately per rating-count bucket
  GBM     : a model that learns the ratio itself from the three predictions + context (user rating count, movie rating count, critic score, etc.)
Only one final score is passed to Stage 2 (same principle as v1/v2).
"""
import pickle

import numpy as np
import pandas as pd

from features_v2 import build_table, meta
from knn import ItemKNN
from models import ALSMF, ContentModel, Stage1Predictor

BLEND_FEATURES = ["mf", "content", "knn", "log_nu", "log_ni", "u_std", "critic", "critic_missing", "year_z"]


def make_stage1(sel, genre_table, seed=42, results_dir=None):
    if sel.get("version") == "v3":
        return Stage1V3(sel, genre_table, seed, results_dir)
    return Stage1Predictor(sel, genre_table, seed=seed)


def blend_features(mf, content, knn, users, items, n_user, n_item, u_std):
    md = meta()
    crit = md.critic.reindex(items).to_numpy()
    year = md.year.reindex(items).to_numpy()
    return pd.DataFrame({
        "mf": mf, "content": content, "knn": knn,
        "log_nu": np.log1p([n_user.get(u, 0) for u in users]),
        "log_ni": np.log1p([n_item.get(m, 0) for m in items]),
        "u_std": [u_std.get(u, np.nan) for u in users],
        "critic": np.nan_to_num(crit, nan=0.6), "critic_missing": np.isnan(crit).astype(float),
        "year_z": np.nan_to_num((year - 1985) / 15.0),
    })


def apply_blender(kind, params, X):
    if kind == "MF only":
        return X.mf.to_numpy()
    if kind == "H2":
        b = np.digitize(np.expm1(X.log_nu.to_numpy()), [50, 200])
        a = np.array([params[str(x)] for x in b])
        return a * X.mf.to_numpy() + (1 - a) * X.content.to_numpy()
    if kind == "LIN3":
        w = params
        return w[0] + w[1] * X.mf.to_numpy() + w[2] * X.content.to_numpy() + w[3] * X.knn.to_numpy()
    if kind == "LIN3_B":
        b = np.digitize(np.expm1(X.log_nu.to_numpy()), [50, 200])
        W = np.array([params[str(x)] for x in b])
        return W[:, 0] + W[:, 1] * X.mf.to_numpy() + W[:, 2] * X.content.to_numpy() + W[:, 3] * X.knn.to_numpy()
    if kind == "GBM":
        return params.predict(X[BLEND_FEATURES])
    raise ValueError(kind)


class Stage1V3:
    def __init__(self, sel, genre_table, seed=42, results_dir=None):
        self.sel, self.gt0, self.seed = sel, genre_table, seed
        self.kind = sel["blender"]
        if self.kind == "GBM":
            with open(results_dir / "stage1_v3_blender.pkl", "rb") as f:
                self.params = pickle.load(f)
        else:
            self.params = sel["blender_params"]

    def fit(self, df):
        U, M, Y = df.user_id.values, df.movie_id.values, df.rating_norm.values
        mf = self.sel["mf"]
        self.mfs = [ALSMF(k=mf["k"], reg=mf["reg"], epochs=mf["epochs"], seed=self.seed + s).fit(U, M, Y)
                    for s in range(mf["n_bag"])]
        c = self.sel["content"]
        self.gt = build_table(df, self.gt0, c["feature_set"], c["kappa"] or 50.0)
        self.content = ContentModel(c["k_mean"], c["lam_user"], c["lam_global"]).fit(U, M, Y, self.gt)
        kn = self.sel["knn"]
        self.knn = ItemKNN(k=kn["k"], sim_shrink=kn["sim_shrink"]).fit(U, M, Y)
        self.n_user = df.groupby("user_id").size().to_dict()
        self.n_item = df.groupby("movie_id").size().to_dict()
        self.u_std = df.groupby("user_id").rating_norm.std().to_dict()
        return self

    def mf_knows_item(self, items):
        return np.array([self.mfs[0].has_item(m) for m in items])

    def components(self, users, items):
        mf = np.mean([m.predict(users, items) for m in self.mfs], axis=0)
        return mf, self.content.predict(users, items), self.knn.predict(users, items)

    def predict(self, users, items):
        users, items = list(users), list(items)
        mf, c, k = self.components(users, items)
        X = blend_features(mf, c, k, users, items, self.n_user, self.n_item, self.u_std)
        return apply_blender(self.kind, self.params, X)
