"""Lerner nur mit numpy: Gruppenmittel, lineare Regression (log), Gradient Boosting (L2 und Quantil). Kein scikit-learn.

Bitgleich zu `hafen-planung/messreihe_verweilzeit/models.py` (dort gegen scikit-learn geprüft: Pinball des Boostings 0,2 bis 1,1 % daneben, lineare Regression identisch). Nicht abgestimmt:
Tiefe 3, Lernrate 0,1, 64 Klassen je Merkmal, Rundenzahl siehe vwz_constants."""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------------------------------------------
def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


# ---------------------------------------------------------------------------------------------------------------
# Einfache Lerner
# ---------------------------------------------------------------------------------------------------------------
class GroupMedian:
    """Nach Containerart (Spalte 0): Median bzw. Mittel je Art."""
    def __init__(self, kind="median"):
        self.kind = kind

    def fit(self, X, y):
        f = np.median if self.kind == "median" else np.mean
        self.m = {a: float(f(y[X[:, 0] == a])) if np.any(X[:, 0] == a) else float(f(y)) for a in range(4)}
        return self

    def predict(self, X):
        return np.array([self.m[int(a)] for a in X[:, 0]])


def design_linear(X):
    art = X[:, 0].astype(int)
    kunde = X[:, 1].astype(int)
    cols = [np.ones(len(X))]
    cols += [(art == a).astype(float) for a in (1, 2, 3)]
    cols += [(kunde == k).astype(float) for k in range(1, 12)]
    cols += [X[:, 2], X[:, 3], X[:, 4], X[:, 5], np.log(np.maximum(X[:, 6], 1e-9)) * (X[:, 6] > 0)]
    return np.column_stack(cols)


class LogLinear:
    """OLS auf log(Verweilzeit) mit Art-, Kunden-Dummies und linearen Zusatzmerkmalen. Streuung: eine Konstante (homoskedastisch)."""
    def __init__(self, ridge=1e-6):
        self.ridge = ridge

    def fit(self, X, y):
        A = design_linear(X)
        ly = np.log(y)
        n, p = A.shape
        self.w = np.linalg.solve(A.T @ A + self.ridge * np.eye(p), A.T @ ly)
        res = ly - A @ self.w
        self.s = float(np.sqrt(np.sum(res ** 2) / max(1, n - p)))
        return self

    def mu(self, X):
        return design_linear(X) @ self.w

    def predict(self, X):            # Median
        return np.exp(self.mu(X))

    def predict_mean(self, X):
        return np.exp(self.mu(X) + 0.5 * self.s ** 2)

    def log_s(self, X):
        return np.full(len(X), self.s)


# ---------------------------------------------------------------------------------------------------------------
# Gradient Boosting (Histogramm-Baeume, Tiefe d)
# ---------------------------------------------------------------------------------------------------------------
class GBT:
    """loss 'l2' (Mittelwert) oder 'quantile' (alpha). Startwert: Mittel bzw. alpha-Quantil. Blattwerte: bei l2 Mittel der Residuen,
    bei quantile das alpha-Quantil der Residuen im Blatt (Linienoptimierung wie ueblich)."""
    def __init__(self, loss="l2", alpha=0.5, n_rounds=150, depth=3, lr=0.1, min_leaf=5, max_bins=64):
        self.loss, self.alpha, self.n_rounds, self.depth, self.lr, self.min_leaf, self.max_bins = loss, alpha, n_rounds, depth, lr, min_leaf, max_bins

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int16)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return Xb

    def fit(self, X, y):
        n, p = X.shape
        self.edges = []
        for j in range(p):
            u = np.unique(X[:, j])
            if len(u) <= self.max_bins:
                e = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([])
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            self.edges.append(e)
        Xb = self._bin(X)
        self.nb = [len(e) + 1 for e in self.edges]
        self.f0 = float(np.mean(y)) if self.loss == "l2" else float(np.quantile(y, self.alpha))
        F = np.full(n, self.f0)
        self.trees = []
        for _ in range(self.n_rounds):
            if self.loss == "l2":
                g = y - F
            else:
                g = np.where(y > F, self.alpha, self.alpha - 1.0)
            tree = self._grow(Xb, g, y - F)
            self.trees.append(tree)
            F += self.lr * self._apply(tree, Xb)
        return self

    def _leaf_value(self, resid):
        if len(resid) == 0:
            return 0.0
        return float(np.mean(resid)) if self.loss == "l2" else float(np.quantile(resid, self.alpha))

    def _grow(self, Xb, g, resid):
        """Baum als Liste von Knoten: ('split', feat, bin_thr, left, right) oder ('leaf', value)."""
        nodes = []

        def build(idx, depth):
            me = len(nodes)
            nodes.append(None)
            if depth < self.depth and len(idx) >= 2 * self.min_leaf:
                best = None
                gi = g[idx]
                tot, cnt = gi.sum(), len(idx)
                base = tot * tot / cnt
                for j in range(Xb.shape[1]):
                    nb = self.nb[j]
                    if nb < 2:
                        continue
                    bj = Xb[idx, j]
                    s = np.bincount(bj, weights=gi, minlength=nb)
                    c = np.bincount(bj, minlength=nb).astype(float)
                    cs, cc = np.cumsum(s)[:-1], np.cumsum(c)[:-1]
                    ok = (cc >= self.min_leaf) & (cnt - cc >= self.min_leaf)
                    if not ok.any():
                        continue
                    gain = np.where(ok, cs * cs / np.maximum(cc, 1) + (tot - cs) ** 2 / np.maximum(cnt - cc, 1) - base, -np.inf)
                    k = int(np.argmax(gain))
                    if best is None or gain[k] > best[0]:
                        best = (gain[k], j, k)
                if best is not None and best[0] > 1e-12:
                    _, j, k = best
                    mask = Xb[idx, j] <= k
                    left = build(idx[mask], depth + 1)
                    right = build(idx[~mask], depth + 1)
                    nodes[me] = ("split", j, k, left, right)
                    return me
            nodes[me] = ("leaf", self._leaf_value(resid[idx]))
            return me

        build(np.arange(len(g)), 0)
        return nodes

    @staticmethod
    def _apply(tree, Xb):
        out = np.empty(len(Xb))
        stack = [(0, np.arange(len(Xb)))]
        while stack:
            node, idx = stack.pop()
            t = tree[node]
            if t[0] == "leaf":
                out[idx] = t[1]
            else:
                _, j, k, l, r = t
                m = Xb[idx, j] <= k
                stack.append((l, idx[m]))
                stack.append((r, idx[~m]))
        return out

    def predict(self, X):
        Xb = self._bin(X)
        F = np.full(len(X), self.f0)
        for tree in self.trees:
            F += self.lr * self._apply(tree, Xb)
        return F
