"""Three Stage 1 individual preference models: MF, Content, Hybrid.

By analogy
- MF (matrix factorization): predicts "people with taste similar to yours liked this movie", compressed into numbers.
  Each person and each movie becomes k 'taste coordinates'; a good match of coordinates gives a high score.
- Content (genre-based): predicts from the movie's genres only, like "you usually like action".
  Even a movie never seen before can be scored if its genres are known (robust to cold start).
- Hybrid: alpha * MF + (1 - alpha) * Content. alpha is chosen on Validation.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------
# 1) Matrix Factorization (with biases, trained by mini-batch SGD)
# ----------------------------------------------------------------------------------------------
class MF:
    """prediction = global mean + user bias + movie bias + (user factors . movie factors)

    hyperparameters: k (number of factors = latent dimension), lr (learning rate), reg (regularization strength), epochs
    """

    def __init__(self, k=32, lr=0.05, reg=0.02, epochs=30, batch=2048, seed=42):
        self.k, self.lr, self.reg, self.epochs, self.batch, self.seed = k, lr, reg, epochs, batch, seed

    def fit(self, users, items, y, val=None, patience=3, verbose=False):
        """Given val=(users, items, y), performance is recorded every epoch and training stops early when it gets worse (early stopping)."""
        rng = np.random.default_rng(self.seed)
        self.uid = {u: i for i, u in enumerate(pd.unique(users))}
        self.iid = {m: i for i, m in enumerate(pd.unique(items))}
        u = np.array([self.uid[x] for x in users])
        it = np.array([self.iid[x] for x in items])
        y = np.asarray(y, float)
        nu, ni, k = len(self.uid), len(self.iid), self.k
        self.mu = y.mean()
        self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        self.P = rng.normal(0, 0.05, (nu, k))
        self.Q = rng.normal(0, 0.05, (ni, k))
        self.history = []
        best = (np.inf, None, 0)
        n = len(y)
        for ep in range(1, self.epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                b = order[s:s + self.batch]
                ub, ib = u[b], it[b]
                pu, qi = self.P[ub], self.Q[ib]
                err = y[b] - (self.mu + self.bu[ub] + self.bi[ib] + np.einsum("ij,ij->i", pu, qi))
                lr, reg = self.lr, self.reg
                np.add.at(self.bu, ub, lr * (err - reg * self.bu[ub]))
                np.add.at(self.bi, ib, lr * (err - reg * self.bi[ib]))
                np.add.at(self.P, ub, lr * (err[:, None] * qi - reg * pu))
                np.add.at(self.Q, ib, lr * (err[:, None] * pu - reg * qi))
            if val is not None:
                score = float(np.sqrt(np.mean((np.asarray(val[2]) - self.predict(val[0], val[1])) ** 2)))
                self.history.append(score)
                if verbose:
                    print(f"  epoch {ep:2d} val RMSE {score:.4f}")
                if score < best[0] - 1e-5:
                    best = (score, (self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy()), ep)
                elif ep - best[2] >= patience:
                    break
        if val is not None and best[1] is not None:
            self.bu, self.bi, self.P, self.Q = best[1]
            self.best_epoch, self.best_val = best[2], best[0]
        return self

    def has_user(self, u):
        return u in self.uid

    def has_item(self, m):
        return m in self.iid

    def predict(self, users, items):
        """Users/movies not seen during training get 0 for that part (= fall back toward the mean)."""
        ui = np.array([self.uid.get(x, -1) for x in users])
        ii = np.array([self.iid.get(x, -1) for x in items])
        out = np.full(len(ui), self.mu)
        ku, ki = ui >= 0, ii >= 0
        out[ku] += self.bu[ui[ku]]
        out[ki] += self.bi[ii[ki]]
        both = ku & ki
        out[both] += np.einsum("ij,ij->i", self.P[ui[both]], self.Q[ii[both]])
        return out

    def fold_in(self, items, y):
        """Compute the factors of a new user not in the model (e.g., MovieTweetings, Beliefs) from their past ratings.
        Movie factors are kept fixed and only that person's bias + factors are solved by ridge regression (same regularization as in training).
        Returns: (bu, p). None if no known movie exists.
        """
        idx = np.array([self.iid.get(m, -1) for m in items])
        keep = idx >= 0
        if keep.sum() == 0:
            return None
        idx, y = idx[keep], np.asarray(y, float)[keep]
        X = np.hstack([np.ones((len(idx), 1)), self.Q[idx]])
        target = y - self.mu - self.bi[idx]
        A = X.T @ X + self.reg * len(idx) * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ target)
        return w[0], w[1:]

    def predict_folded(self, user_vec, items):
        bu, p = user_vec
        idx = np.array([self.iid.get(m, -1) for m in items])
        out = np.full(len(idx), self.mu + bu)
        k = idx >= 0
        out[k] += self.bi[idx[k]] + self.Q[idx[k]] @ p
        return out


def recency_weights(timestamps, half_life_days):
    """Ratings older by one half-life get half the weight. If None, all weights are 1 (same as version 1)."""
    if not half_life_days:
        return None
    ts = np.asarray(timestamps, float)
    age_days = (ts.max() - ts) / 86400.0
    return 0.5 ** (age_days / half_life_days)


class ALSMF(MF):
    """Version that trains the same MF formula with 'alternating least squares (ALS)' (additional candidate).

    SGD results can fluctuate with the learning rate and order. ALS alternately solves the user side / movie side 'exactly',
    so it is more stable. Regularization is proportional to the number of ratings (reg * n). k=0 is the baseline that uses 'biases only, no taste factors'.
    """

    def __init__(self, k=32, reg=0.05, epochs=15, seed=42):
        super().__init__(k=k, lr=0.0, reg=reg, epochs=epochs, seed=seed)

    @staticmethod
    def _index(codes, n):
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(n + 1))
        return order, bounds

    def _solve_side(self, own_n, own_order, own_bounds, other_idx, other_F, other_b, y, wts=None):
        k = other_F.shape[1]
        out_b, out_F = np.zeros(own_n), np.zeros((own_n, k))
        eye = np.eye(k + 1)
        for j in range(own_n):
            rows = own_order[own_bounds[j]:own_bounds[j + 1]]
            if len(rows) == 0:
                continue
            o = other_idx[rows]
            X = np.hstack([np.ones((len(rows), 1)), other_F[o]])
            t = y[rows] - self.mu - other_b[o]
            if wts is None:
                w = np.linalg.solve(X.T @ X + self.reg * len(rows) * eye, X.T @ t)
            else:  # larger weight on recent ratings (version 2 candidate)
                ww = wts[rows]
                w = np.linalg.solve(X.T @ (X * ww[:, None]) + self.reg * ww.sum() * eye, X.T @ (t * ww))
            out_b[j], out_F[j] = w[0], w[1:]
        return out_b, out_F

    def fit(self, users, items, y, val=None, patience=2, verbose=False, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        self.uid = {u: i for i, u in enumerate(pd.unique(users))}
        self.iid = {m: i for i, m in enumerate(pd.unique(items))}
        u = np.array([self.uid[x] for x in users])
        it = np.array([self.iid[x] for x in items])
        y = np.asarray(y, float)
        nu, ni, k = len(self.uid), len(self.iid), self.k
        wts = None if sample_weight is None else np.asarray(sample_weight, float)
        self.mu = y.mean() if wts is None else float(np.average(y, weights=wts))
        self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        self.P = rng.normal(0, 0.05, (nu, k))
        self.Q = rng.normal(0, 0.05, (ni, k))
        uo, ub = self._index(u, nu)
        io, ib = self._index(it, ni)
        self.history, best = [], (np.inf, None, 0)
        for ep in range(1, self.epochs + 1):
            self.bu, self.P = self._solve_side(nu, uo, ub, it, self.Q, self.bi, y, wts)
            self.bi, self.Q = self._solve_side(ni, io, ib, u, self.P, self.bu, y, wts)
            if val is not None:
                score = float(np.sqrt(np.mean((np.asarray(val[2]) - self.predict(val[0], val[1])) ** 2)))
                self.history.append(score)
                if verbose:
                    print(f"  iter {ep:2d} val RMSE {score:.4f}")
                if score < best[0] - 1e-5:
                    best = (score, (self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy()), ep)
                elif ep - best[2] >= patience:
                    break
        if val is not None and best[1] is not None:
            self.bu, self.bi, self.P, self.Q = best[1]
            self.best_epoch, self.best_val = best[2], best[0]
        return self


# ----------------------------------------------------------------------------------------------
# 2) Content-Based (genre taste model)
# ----------------------------------------------------------------------------------------------
class ContentModel:
    """prediction = user mean + (the person's per-genre preference weights . the movie's genre composition)

    - the user mean is shrunk toward the global mean when there are few ratings (k_mean).
    - genre weights start from the 'genre effect shared by everyone' and move toward the individual as personal data grows (lam_user).
    The movie ID itself is never used -> any new movie with genres can be scored.
    """

    def __init__(self, k_mean=5.0, lam_user=10.0, lam_global=1.0):
        self.k_mean, self.lam_user, self.lam_global = k_mean, lam_user, lam_global

    def fit(self, users, items, y, genre_table):
        df = pd.DataFrame({"u": users, "m": items, "y": np.asarray(y, float)})
        self.gt = genre_table
        self.mu = df.y.mean()
        m_codes, m_uniq = pd.factorize(df.m)                 # build the per-movie vectors once and expand (for large data; same result)
        X = np.vstack([genre_table[m] for m in m_uniq])[m_codes]
        st = df.groupby("u").y.agg(["sum", "count"])
        self.user_mean = ((st["sum"] + self.k_mean * self.mu) / (st["count"] + self.k_mean)).to_dict()
        base = df.u.map(self.user_mean).to_numpy()
        resid = df.y.to_numpy() - base
        d = X.shape[1]
        self.w_global = np.linalg.solve(X.T @ X + self.lam_global * np.eye(d), X.T @ resid)
        resid2 = resid - X @ self.w_global
        self.w_user = {}
        codes, uniq = pd.factorize(df.u)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        for j, user in enumerate(uniq):
            rows = order[bounds[j]:bounds[j + 1]]
            Xu, ru = X[rows], resid2[rows]
            self.w_user[user] = self.w_global + np.linalg.solve(Xu.T @ Xu + self.lam_user * np.eye(d), Xu.T @ ru)
        return self

    def profile_from_history(self, items, y, genre_table=None):
        """Compute a new user's profile (mean, genre weights) from past ratings only. Model parameters (mu, w_global) are kept fixed."""
        gt = genre_table or self.gt
        y = np.asarray(y, float)
        if len(y) == 0:
            return self.mu, self.w_global.copy()
        mean = (y.sum() + self.k_mean * self.mu) / (len(y) + self.k_mean)
        X = np.vstack([gt.get(m, np.zeros(len(self.w_global))) for m in items])
        r = y - mean - X @ self.w_global
        d = X.shape[1]
        w = self.w_global + np.linalg.solve(X.T @ X + self.lam_user * np.eye(d), X.T @ r)
        return mean, w

    def predict(self, users, items, genre_table=None):
        gt = genre_table or self.gt
        zero = np.zeros(len(self.w_global))
        out = np.empty(len(users))
        for n, (u, m) in enumerate(zip(users, items)):
            mean = self.user_mean.get(u, self.mu)
            w = self.w_user.get(u, self.w_global)
            out[n] = mean + gt.get(m, zero) @ w
        return out

    def predict_profile(self, profile, items, genre_table=None):
        gt = genre_table or self.gt
        mean, w = profile
        zero = np.zeros(len(w))
        return np.array([mean + gt.get(m, zero) @ w for m in items])


# ----------------------------------------------------------------------------------------------
# 4) Stage 1 final predictor: trained with exactly the settings chosen on Validation, it outputs only 'one individual preference score'
# ----------------------------------------------------------------------------------------------
class Stage1Predictor:
    """Train MF and Content with the settings in selection (stage1_selection_final.json) and mix them with the chosen structure.
    Only this class's final score is passed to Stage 2 (MF factors and genre weights are not passed on).

    structure: 'MF only' | 'Content only' | 'H1_global'(alpha) | 'H2_bucket'(alpha by rating count) | 'H3_stack'(a+b*MF+c*C)
    """

    def __init__(self, selection, genre_table, seed=42):
        self.sel, self.gt, self.structure = selection, genre_table, selection["structure"]
        mf, c = selection["mf"], selection["content"]
        self.use_mf = self.structure != "Content only"
        self.use_content = self.structure != "MF only"
        if self.use_mf:
            self.mf = (ALSMF(k=mf["k"], reg=mf["reg"], epochs=mf["epochs"], seed=seed) if mf["type"] == "als"
                       else MF(k=mf["k"], lr=mf["lr"], reg=mf["reg"], epochs=mf["epochs"], seed=seed))
        if self.use_content:
            self.content = ContentModel(c["k_mean"], c["lam_user"], c["lam_global"])

    def fit(self, df):
        U, M, Y = df.user_id.values, df.movie_id.values, df.rating_norm.values
        if self.use_mf:
            # epochs are fixed to the value chosen on Validation (no early stopping)
            wts = recency_weights(df.timestamp.values, self.sel["mf"].get("half_life"))
            if isinstance(self.mf, ALSMF):
                self.mf.fit(U, M, Y, sample_weight=wts)
            else:
                self.mf.fit(U, M, Y)
        if self.use_content:
            fs = self.sel["content"].get("feature_set", "G")
            if fs != "G":   # version 2: table of genre + year/critic/director features
                from features_v2 import build_table
                self.gt = build_table(df, self.gt, fs, self.sel["content"].get("kappa", 50.0))
            self.content.fit(U, M, Y, self.gt)
        self.n_hist = df.groupby("user_id").size().to_dict()
        return self

    def _combine(self, mfp, cp, n_hist):
        s = self.structure
        if s == "MF only":
            return mfp
        if s == "Content only":
            return cp
        if s == "H1_global":
            a = self.sel["alpha"]
            return a * mfp + (1 - a) * cp
        if s == "H2_bucket":
            edges = self.sel["bucket_edges"]
            b = np.digitize(np.asarray(n_hist, float), edges[1:])
            a = np.array([self.sel["hybrid_params"][str(x)] for x in b])
            return a * mfp + (1 - a) * cp
        w = self.sel["hybrid_params"]
        return w[0] + w[1] * mfp + w[2] * cp

    def mf_knows_item(self, items):
        if not self.use_mf:
            return np.ones(len(items), bool)
        return np.array([self.mf.has_item(m) for m in items])

    def predict(self, users, items):
        """For users inside the model (MovieLens)."""
        n = len(users)
        mfp = self.mf.predict(users, items) if self.use_mf else np.zeros(n)
        cp = self.content.predict(users, items) if self.use_content else np.zeros(n)
        return self._combine(mfp, cp, [self.n_hist.get(u, 0) for u in users])

    def predict_new_user(self, hist_items, hist_y, items, genre_table=None, allow_content_fallback=False):
        """For users outside the model (MovieTweetings/Beliefs): fold in from past ratings, then predict.
        Returns: (prediction, whether MF was usable)
        - if MF cannot handle that person/movie: fall back to Content only when allow_content_fallback=True, otherwise NaN.
        """
        gt = genre_table or self.gt
        fs = self.sel["content"].get("feature_set", "G") if self.use_content else "G"
        if genre_table is not None and fs != "G":
            from features_v2 import extend_with_genres
            gt = extend_with_genres(self.gt, genre_table, fs)
        n = len(items)
        cp = self.content.predict_profile(self.content.profile_from_history(hist_items, hist_y, gt), items, gt) \
            if self.use_content else np.zeros(n)
        if not self.use_mf:
            return cp, np.ones(n, bool)
        vec = self.mf.fold_in(hist_items, hist_y)
        known = self.mf_knows_item(items) & (vec is not None)
        mfp = self.mf.predict_folded(vec, items) if vec is not None else np.zeros(n)
        # the rating-count bucket (H2_bucket) reflects 'how much MF knows about the person', so it counts past ratings of movies known to MF
        n_mf_known = int(sum(self.mf.has_item(m) for m in hist_items))
        out = self._combine(mfp, cp, np.full(n, n_mf_known))
        fallback = cp if (allow_content_fallback and self.use_content) else np.full(n, np.nan)
        return np.where(known, out, fallback), known


# ----------------------------------------------------------------------------------------------
# 3) Hybrid
# ----------------------------------------------------------------------------------------------
def hybrid(mf_pred, content_pred, alpha, mf_available=None):
    """alpha * MF + (1-alpha) * Content.
    Cells with mf_available=False (MF does not know the movie = cold start) use the Content prediction only."""
    mf_pred, content_pred = np.asarray(mf_pred, float), np.asarray(content_pred, float)
    out = alpha * mf_pred + (1 - alpha) * content_pred
    if mf_available is not None:
        mf_available = np.asarray(mf_available, bool)
        out = np.where(mf_available, out, content_pred)
    return out
