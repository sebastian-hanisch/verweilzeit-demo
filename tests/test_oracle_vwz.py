"""Unabhängige Orakel für die Rechenkerne der Verweilzeit-Demo (ohne scikit-learn, nur eigene einfache Neuimplementierungen):
(1) Gradient Boosting (L2 und Quantil) gegen eine schleifenbasierte Neuimplementierung aus der Spezifikation (Histogramm-Klassen, beste Aufteilung nach
Gewinn, Blattwert Mittel bzw. Quantil der Residuen); Instanzen mit exaktem Gleichstand der Aufteilungsgewinne (Gleitkommarauschen entscheidet) werden übersprungen;
(2) Einlagern und Umstapeln (Bestfit, Niedrigster Stapel, gemischt) gegen eine eigene Simulation mit Zustandsfolge; (3) Ankunftszeiten des Generators aus den
Verweilzeiten mit einer Liste statt eines Heaps neu abgeleitet, Merkmale und bedingte Verteilung von Hand; (4) Bayes-Orakel gegen ein numerisch integriertes Posterior."""

import math
import random

import numpy as np
import pytest

import vwz_forecast as F
import vwz_generator as G
import vwz_models as M
import vwz_stk_core as K


# --- (1) Gradient Boosting ---------------------------------------------------------------------------------------------------------------------


def _quant(vals, q):
    v = sorted(vals)
    pos = (len(v) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _ref_gbt(X, y, loss, alpha, rounds, depth, lr, min_leaf, max_bins):
    """Gibt (Vorhersagefunktion, mehrdeutig) zurück; mehrdeutig = ein Gleichstand der Gewinne (< 1e-9) mit anderer Aufteilung kam vor."""
    n, p = X.shape
    cols = [[float(X[i, j]) for i in range(n)] for j in range(p)]
    edges = []
    for j in range(p):
        u = sorted(set(cols[j]))
        if len(u) <= max_bins:
            edges.append([(u[k] + u[k + 1]) / 2 for k in range(len(u) - 1)])
        else:
            edges.append(sorted({_quant(cols[j], k / max_bins) for k in range(1, max_bins)}))

    def binof(j, x):
        return sum(1 for e in edges[j] if e <= x)

    B = [[binof(j, cols[j][i]) for j in range(p)] for i in range(n)]
    nb = [len(e) + 1 for e in edges]
    f0 = sum(y) / n if loss == "l2" else _quant(list(y), alpha)
    F_ = [f0] * n
    trees = []
    flag = {"amb": False}
    for _ in range(rounds):
        g = [y[i] - F_[i] for i in range(n)] if loss == "l2" else [alpha if y[i] > F_[i] else alpha - 1.0 for i in range(n)]
        resid = [y[i] - F_[i] for i in range(n)]

        def leaf(idx):
            r = [resid[i] for i in idx]
            if not r:
                return 0.0
            return sum(r) / len(r) if loss == "l2" else _quant(r, alpha)

        def grow(idx, d):
            if d < depth and len(idx) >= 2 * min_leaf:
                tot = sum(g[i] for i in idx)
                cnt = len(idx)
                gains = []
                for j in range(p):
                    for k in range(nb[j] - 1):
                        left = [i for i in idx if B[i][j] <= k]
                        cl, cr = len(left), cnt - len(left)
                        if cl < min_leaf or cr < min_leaf:
                            continue
                        sl = sum(g[i] for i in left)
                        gains.append((sl * sl / cl + (tot - sl) ** 2 / cr - tot * tot / cnt, j, k))
                if gains:
                    top = max(x[0] for x in gains)
                    if top > 1e-12:
                        winners = [x for x in gains if x[0] >= top - 1e-9]
                        if len(winners) > 1:
                            flag["amb"] = True
                        _, j, k = winners[0]
                        return ("s", j, k, grow([i for i in idx if B[i][j] <= k], d + 1), grow([i for i in idx if B[i][j] > k], d + 1))
            return ("l", leaf(idx))

        tree = grow(list(range(n)), 0)
        trees.append(tree)
        for i in range(n):
            t = tree
            while t[0] == "s":
                t = t[3] if B[i][t[1]] <= t[2] else t[4]
            F_[i] += lr * t[1]

    def predict(Xn):
        out = []
        for row in Xn:
            bb = [binof(j, float(row[j])) for j in range(p)]
            v = f0
            for t in trees:
                while t[0] == "s":
                    t = t[3] if bb[t[1]] <= t[2] else t[4]
                v += lr * t[1]
            out.append(v)
        return np.array(out)

    return predict, flag


def test_gradient_boosting_against_a_loop_based_reimplementation():
    rng = np.random.default_rng(2606)
    pr = random.Random(5)
    compared = 0
    for _ in range(150):
        n, p = int(rng.integers(12, 60)), int(rng.integers(1, 5))
        cols = []
        for _j in range(p):
            kind = pr.choice(["cont", "int", "bin", "const"])
            cols.append({"cont": lambda: rng.normal(0, 1, n), "int": lambda: rng.integers(0, 5, n).astype(float),
                         "bin": lambda: rng.integers(0, 2, n).astype(float), "const": lambda: np.full(n, 3.0)}[kind]())
        X = np.column_stack(cols)
        y = X[:, 0] * 1.5 + rng.normal(0, 1, n) + (X[:, -1] > 0) * 2
        loss, alpha = pr.choice(["l2", "quantile"]), pr.choice([0.1, 0.5, 0.7, 0.9])
        rounds, depth, lr = pr.choice([1, 3, 8]), pr.choice([0, 1, 2, 3]), pr.choice([0.1, 0.5, 1.0])
        ml, mb = pr.choice([1, 3, 5]), pr.choice([4, 8, 64])
        Xn = np.vstack([X, X + rng.normal(0, 0.7, X.shape)])
        ref, flag = _ref_gbt(X, list(y), loss, alpha, rounds, depth, lr, ml, mb)
        if flag["amb"]:
            continue
        got = M.GBT(loss, alpha, n_rounds=rounds, depth=depth, lr=lr, min_leaf=ml, max_bins=mb).fit(X, y).predict(Xn)
        assert np.allclose(got, ref(Xn), atol=1e-9)
        compared += 1
    assert compared >= 60


# --- (2) Einlagern und Umstapeln -------------------------------------------------------------------------------------------------------------


def _sim(events, H, nst, est, place, reloc):
    stacks = [[] for _ in range(nst)]
    moves, states = 0, []

    def choose(kind, c, excl):
        cand = [i for i in range(nst) if i != excl and len(stacks[i]) < H]
        if kind == "low":
            return min(cand, key=lambda i: (len(stacks[i]), i))

        def key(i):
            top = est[stacks[i][-1]] if stacks[i] else math.inf
            return (0, top, i) if top >= est[c] else (1, -top, i)
        return min(cand, key=key)

    for kd, c in events:
        if kd == "A":
            stacks[choose(place, c, None)].append(c)
        else:
            x = next(i for i in range(nst) if c in stacks[i])
            while stacks[x][-1] != c:
                b = stacks[x].pop()
                stacks[choose(reloc, b, x)].append(b)
                moves += 1
            stacks[x].pop()
        states.append(tuple(tuple(s) for s in stacks))
    return moves, states


def test_placement_and_relocation_against_an_own_simulation():
    rng = random.Random(3)
    for it in range(40):
        S, H, fp, nc = rng.randint(3, 8), rng.randint(3, 6), rng.choice([40, 60, 80, 100]), rng.choice([60, 100])
        s = G.generate(S, H, fp / 100, nc, rng.randrange(10000), nu=rng.choice([None, 0.6]))
        est = F.noisy_est(s, rng.choice([0.0, 0.25, 0.75, 1.5]), it)
        for place, reloc, rule, rl in (("bf", "bf", K.bestfit, K.bestfit), ("low", "low", K.niedrigster_stapel, K.niedrigster_stapel),
                                       ("low", "bf", K.niedrigster_stapel, K.bestfit)):
            r = K.run(s.inst, est, rule, reloc_rule=rl, record=True)
            moves, states = _sim(s.inst.events, H, S, est, place, reloc)
            assert moves == r.moves and tuple(states) == tuple(st.stacks for st in r.steps)


# --- (3) Generator ------------------------------------------------------------------------------------------------------------------------------


def test_generator_arrival_times_and_conditional_distribution_rederived():
    rng = random.Random(8)
    for _ in range(25):
        S, H, fp, nc = rng.randint(3, 8), rng.randint(3, 6), rng.choice([40, 60, 80, 100]), rng.choice([60, 100])
        seed, rho = rng.randrange(10000), rng.choice([0.7, 3.0])
        s = G.generate(S, H, fp / 100, nc, seed, rho=rho)
        cap = max(1, round(fp / 100 * (S - 1) * H))
        lam = rho * cap / G.marginal_mean_dwell(1.0)
        E_ = np.random.default_rng([seed, 3]).exponential(1.0, nc)
        present, tp, ta = [], 0.0, []
        for k in range(nc):
            t = tp + E_[k] / lam
            present = [d for d in present if d > t]
            if len(present) >= cap:
                dmin = min(present)
                t = max(t, dmin)
                present.remove(dmin)
            present.append(t + s.dwell[k])
            ta.append(t)
            tp = t
        assert np.allclose(ta, s.t_arr, rtol=0, atol=1e-9)
        U = np.random.default_rng([seed, 1]).random((nc, 6))
        Z = np.random.default_rng([seed, 2]).standard_normal((nc, 3))
        for k in range(nc):
            u = U[k]
            art = 0 if u[0] < 0.35 else 1 if u[0] < 0.65 else 2 if u[0] < 0.90 else 3
            kunde, zoll, we = int(u[1] * 12), int(art == 0 and u[2] < 0.25), int(int(s.t_arr[k]) % 7 >= 5)
            vor = 1 + 9 * u[3] if art == 1 else (0.5 + 5.5 * u[3] if art == 2 else 0.0)
            assert tuple(s.X[k, :4]) == (art, kunde, zoll, we) and s.X[k, 4] == pytest.approx(vor, abs=1e-12)
            med, sc, sb = {0: (3.5 * math.exp(0.7 * zoll + 0.30 * we), 1.0, 0.55), 1: (0.6 + 0.85 * vor, 0.3, 0.25),
                           2: (0.3 + 0.8 * vor, 0.3, 0.35), 3: (7.0 * math.exp(0.30 * we), 1.0, 0.90)}[art]
            mu = math.log(med) + G.UK[kunde] * sc
            assert s.mu[k] == pytest.approx(mu, abs=1e-12) and s.s[k] == pytest.approx(sb, abs=1e-12)
            assert s.dwell[k] == pytest.approx(math.exp(mu + sb * Z[k, 0]), rel=1e-9)


# --- (4) Bayes-Orakel ---------------------------------------------------------------------------------------------------------------------------


def test_bayes_oracle_against_a_numerically_integrated_posterior():
    rb = random.Random(2)
    grid = np.linspace(-10, 14, 100001)
    for _ in range(30):
        mu, s, nu, y = rb.uniform(0, 3), rb.uniform(0.2, 1.0), rb.choice([0.15, 0.3, 0.6, 1.0]), rb.uniform(-1, 5)
        m, sd = G.bayes_oracle(np.array([mu]), np.array([s]), np.array([math.exp(y)]), nu)
        lp = -0.5 * ((grid - mu) / s) ** 2 - 0.5 * ((y - grid) / nu) ** 2
        w = np.exp(lp - lp.max())
        w /= w.sum()
        mean = float((w * grid).sum())
        assert m[0] == pytest.approx(mean, abs=1e-6)
        assert sd[0] == pytest.approx(math.sqrt(float((w * (grid - mean) ** 2).sum())), abs=1e-6)
