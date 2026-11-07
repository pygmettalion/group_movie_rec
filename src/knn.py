"""[version 3] Item-kNN collaborative filtering: "what did this person give to movies similar to this one?"

1) baseline = global mean + user bias + movie bias (shrunk toward 0 when there are few ratings: shrink)
2) movie-movie similarity = cosine similarity of the residuals after removing the baseline (the taste part); smaller when few people rated both (sim_shrink)
3) prediction = baseline + (similarity-weighted mean of the 'residuals' of the k most similar movies this person has watched)
"""
import numpy as np
import pandas as pd
from scipy import sparse


class ItemKNN:
    def __init__(self, k=40, shrink=25.0, sim_shrink=50.0):
        self.k, self.shrink, self.sim_shrink = k, shrink, sim_shrink

    def fit(self, users, items, y):
        df = pd.DataFrame({"u": users, "m": items, "y": np.asarray(y, float)})
        self.mu = df.y.mean()
        r = df.y - self.mu
        bi = r.groupby(df.m).agg(["sum", "count"])
        self.bi = (bi["sum"] / (bi["count"] + self.shrink)).to_dict()
        r2 = r - df.m.map(self.bi)
        bu = r2.groupby(df.u).agg(["sum", "count"])
        self.bu = (bu["sum"] / (bu["count"] + self.shrink)).to_dict()
        res = (r2 - df.u.map(self.bu)).to_numpy()
        self.uid = {u: i for i, u in enumerate(pd.unique(df.u))}
        self.iid = {m: i for i, m in enumerate(pd.unique(df.m))}
        ui = df.u.map(self.uid).to_numpy()
        ii = df.m.map(self.iid).to_numpy()
        R = sparse.csr_matrix((res, (ui, ii)), shape=(len(self.uid), len(self.iid)))
        B = sparse.csr_matrix((np.ones(len(res)), (ui, ii)), shape=R.shape)
        num = (R.T @ R).toarray()
        sq = np.sqrt(np.maximum(np.diag(num), 1e-12))
        co = (B.T @ B).toarray()
        S = num / np.outer(sq, sq) * (co / (co + self.sim_shrink))
        np.fill_diagonal(S, 0.0)
        self.S = S.astype(np.float32)
        # per user: (indices of watched movies, residual values)
        order = np.argsort(ui, kind="stable")
        bounds = np.searchsorted(ui[order], np.arange(len(self.uid) + 1))
        self.hist = {u: (ii[order[bounds[j]:bounds[j + 1]]], res[order[bounds[j]:bounds[j + 1]]])
                     for u, j in self.uid.items()}
        return self

    def baseline(self, u, m):
        return self.mu + self.bu.get(u, 0.0) + self.bi.get(m, 0.0)

    def _neighbor_part(self, hist_idx, hist_res, q_idx):
        if len(hist_idx) == 0:
            return np.zeros(len(q_idx))
        sims = self.S[np.ix_(q_idx, hist_idx)]
        k = min(self.k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if sims.shape[1] > k else np.tile(np.arange(sims.shape[1]), (len(q_idx), 1))
        s = np.take_along_axis(sims, top, 1)
        s = np.maximum(s, 0.0)                     # only similar movies (movies with opposite taste are ignored)
        v = hist_res[top]
        den = s.sum(1)
        return np.where(den > 0, (s * v).sum(1) / np.maximum(den, 1e-9), 0.0)

    def predict(self, users, items):
        users, items = np.asarray(users), np.asarray(items)
        out = np.array([self.baseline(u, m) for u, m in zip(users, items)])
        df = pd.DataFrame({"u": users, "m": items, "pos": np.arange(len(users))})
        for u, d in df.groupby("u"):
            if u not in self.hist:
                continue
            known = d[d.m.isin(self.iid)]
            if len(known) == 0:
                continue
            q = known.m.map(self.iid).to_numpy()
            hi, hr = self.hist[u]
            out[known.pos.to_numpy()] += self._neighbor_part(hi, hr, q)
        return out
