"""Die Lerner: unabhängige Nachrechnung (lstsq), Handfälle, Determinismus, Leckage-Sonde, Gegenprobe gegen scikit-learn (nur wenn installiert)."""

import numpy as np
import pytest

import vwz_generator as G
import vwz_models as M


def test_pinball_hand_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.pinball(y, np.full(4, 2.5), 0.5) == pytest.approx(0.5 * np.mean(np.abs(y - 2.5)))
    assert M.pinball(y, np.full(4, 0.0), 0.9) == pytest.approx(0.9 * 2.5)             # Unterschätzung wiegt alpha
    assert M.pinball(y, np.full(4, 10.0), 0.9) == pytest.approx(0.1 * 7.5)            # Überschätzung wiegt 1 - alpha
    assert M.pinball(y, y, 0.7) == 0.0


def test_group_median_by_art_with_fallback():
    X = np.array([[0.0, 0], [0, 0], [1, 0], [1, 0], [1, 0]])
    y = np.array([1.0, 3.0, 2.0, 4.0, 9.0])
    med = M.GroupMedian("median").fit(X, y)
    mean = M.GroupMedian("mean").fit(X, y)
    assert med.m[0] == 2.0 and med.m[1] == 4.0 and mean.m[0] == 2.0 and mean.m[1] == 5.0
    assert med.m[2] == med.m[3] == 3.0 and mean.m[2] == mean.m[3] == pytest.approx(3.8)   # fehlende Art: Gesamtwert
    assert mean.predict(np.array([[1.0, 0], [0, 0], [1, 0]])).tolist() == [5.0, 2.0, 5.0]


def test_design_matrix_columns():
    X = np.array([[2.0, 3, 1, 1, 4.5, -0.5, 6.0], [0.0, 0, 0, 0, 0.0, 0.2, 0.0]])
    A = M.design_linear(X)
    assert A.shape == (2, 1 + 3 + 11 + 5) and A[:, 0].tolist() == [1, 1]
    assert A[0, 1:4].tolist() == [0, 1, 0] and A[1, 1:4].tolist() == [0, 0, 0]              # Art-Dummies gegen Import
    assert A[0, 4:15].tolist() == [0, 0, 1] + [0] * 8 and A[1, 4:15].tolist() == [0] * 11   # Kunden-Dummies gegen Kunde 0
    assert A[0, 15:].tolist() == [1, 1, 4.5, -0.5, pytest.approx(np.log(6.0))] and A[1, 15:].tolist() == [0, 0, 0.0, 0.2, 0.0]   # ohne Ankündigung kein Log-Term


def test_loglinear_equals_an_independent_least_squares():
    X, y = G.make_history(2000, 3, nu=0.6)
    ll = M.LogLinear(ridge=0.0).fit(X, y)
    w, *_ = np.linalg.lstsq(M.design_linear(X), np.log(y), rcond=None)
    assert np.allclose(ll.w, w, atol=1e-6)
    res = np.log(y) - M.design_linear(X) @ w
    assert ll.s == pytest.approx(np.sqrt(np.sum(res ** 2) / (len(y) - len(w))), rel=1e-9)
    assert np.allclose(ll.mu(X), M.design_linear(X) @ w, atol=1e-6) and np.allclose(ll.predict(X), np.exp(ll.mu(X)))
    assert np.allclose(ll.predict_mean(X), np.exp(ll.mu(X) + ll.s ** 2 / 2)) and ll.log_s(X[:3]).tolist() == [ll.s] * 3


def test_loglinear_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 20000
    X = np.zeros((n, 7))
    X[:, 0] = rng.integers(0, 4, n)
    X[:, 1] = rng.integers(0, 12, n)
    X[:, 4] = rng.uniform(0, 5, n)
    y = np.exp(1.0 + 0.5 * (X[:, 0] == 2) + 0.2 * X[:, 4] + 0.1 * rng.standard_normal(n))
    ll = M.LogLinear().fit(X, y)
    assert ll.w[0] == pytest.approx(1.0, abs=0.02) and ll.w[2] == pytest.approx(0.5, abs=0.02) and ll.w[17] == pytest.approx(0.2, abs=0.01) and ll.s == pytest.approx(0.1, abs=0.005)


def test_gbt_hand_cases():
    Xg = np.array([[0.0]] * 50 + [[1.0]] * 50)
    yg = np.array([1.0] * 50 + [5.0] * 50)
    probe = np.array([[0.0], [1.0]])
    assert np.allclose(M.GBT("l2", n_rounds=60, depth=2).fit(Xg, yg).predict(probe), [1.0, 5.0], atol=0.05)
    assert np.allclose(M.GBT("quantile", 0.5, n_rounds=60, depth=2).fit(Xg, yg).predict(probe), [1.0, 5.0], atol=0.05)
    rng = np.random.default_rng(2)
    Xc = np.zeros((4000, 1))
    yc = rng.exponential(2.0, 4000)
    q90 = M.GBT("quantile", 0.9, n_rounds=80, depth=1).fit(Xc, yc).predict(np.zeros((1, 1)))[0]
    assert q90 == pytest.approx(np.quantile(yc, 0.9), abs=0.15)
    assert M.GBT("l2", n_rounds=120, depth=1).fit(Xc, yc).predict(np.zeros((1, 1)))[0] == pytest.approx(yc.mean(), abs=0.01)


def test_gbt_depth_zero_is_a_constant_and_learning_rate_sets_the_pace():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.where(X[:, 0] < 20, 0.0, 10.0)
    flat = M.GBT("l2", n_rounds=5, depth=0).fit(X, y).predict(X)
    assert np.allclose(flat, flat[0]) and flat[0] == pytest.approx(5.0)                 # ohne Splits nur der Startwert (Mittel)
    one = M.GBT("l2", n_rounds=1, depth=1, lr=0.1, min_leaf=5).fit(X, y).predict(X)
    assert one[0] == pytest.approx(5.0 - 0.1 * 5.0) and one[-1] == pytest.approx(5.0 + 0.1 * 5.0)    # eine Runde: Start plus lr mal Blattmittel der Residuen
    full = M.GBT("l2", n_rounds=100, depth=1, lr=0.1, min_leaf=5).fit(X, y).predict(X)
    assert np.allclose(full, y, atol=0.01)


def test_gbt_starting_value_is_the_mean_for_l2_and_the_quantile_otherwise():
    y = np.array([1.0, 2.0, 10.0])
    X = np.zeros((3, 1))
    assert M.GBT("l2", n_rounds=1, depth=0).fit(X, y).predict(X)[0] == pytest.approx(13.0 / 3)         # Mittel, nicht Median (2)
    assert M.GBT("quantile", 0.5, n_rounds=0, depth=0).fit(X, y).predict(X)[0] == 2.0 and M.GBT("quantile", 0.9, n_rounds=0).fit(X, y).predict(X)[0] == pytest.approx(np.quantile(y, 0.9))


def test_gbt_uses_all_distinct_values_as_bins_when_they_fit():
    X = np.array([0.0] * 50 + [1, 2, 3, 4, 5, 6] + [7.0] * 50).reshape(-1, 1)                          # genau 8 verschiedene Werte, stark schief
    assert M.GBT("l2", n_rounds=1, max_bins=8).fit(X, np.arange(106.0)).nb[0] == 8                    # Mittelpunkte statt Quantile (die würden Klassen verschmelzen)


def test_gbt_respects_the_minimum_leaf_size_and_ignores_constant_features():
    X = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
    y = np.array([0.0] * 2 + [10.0] * 18)
    model = M.GBT("l2", n_rounds=1, depth=1, lr=1.0, min_leaf=5).fit(X, y)
    assert len(set(model.predict(X).round(6))) == 2 and min(sum(model.predict(X) == v) for v in set(model.predict(X))) >= 5      # nie ein Blatt unter 5 Beispielen
    assert model.nb == [20, 1]                                                          # konstante Spalte: eine Klasse, wird nicht geteilt
    binned = M.GBT("l2", n_rounds=1, depth=1, max_bins=8).fit(np.arange(200, dtype=float).reshape(-1, 1), np.arange(200.0))
    assert binned.nb[0] <= 8


def test_gbt_is_deterministic_and_quantiles_are_ordered():
    X, y = G.make_history(500, 4, nu=0.6)
    ly = np.log(y)
    a = M.GBT("quantile", 0.7, n_rounds=40).fit(X, ly)
    b = M.GBT("quantile", 0.7, n_rounds=40).fit(X, ly)
    assert np.array_equal(a.predict(X), b.predict(X))
    lo, hi = (M.GBT("quantile", q, n_rounds=40).fit(X, ly).predict(X).mean() for q in (0.3, 0.9))
    assert lo < a.predict(X).mean() < hi
    frac = np.mean(ly <= M.GBT("quantile", 0.9, n_rounds=80).fit(X, ly).predict(X))
    assert frac == pytest.approx(0.9, abs=0.05)                                         # Anteil unter dem 0,9-Quantil im Training


def test_leakage_probe_shuffled_targets_learn_nothing():
    """Mit vertauschten Zielgrößen ist das Modell nicht besser als die Konstante (Pinball >= 97 % der Konstanten)."""
    rng = np.random.default_rng(0)
    Xtr, ytr = G.make_history(1000, 7, nu=0.6)
    ysh = rng.permutation(ytr)
    Xte, yte = G.make_history(20000, 8, nu=0.6)
    g = M.GBT("quantile", 0.5).fit(Xtr, np.log(ysh))
    assert M.pinball(yte, np.exp(g.predict(Xte)), 0.5) > 0.97 * M.pinball(yte, np.full(len(yte), np.median(ysh)), 0.5)


def test_the_learners_learn_the_signal_when_it_is_there():
    """Gegenstück zur Leckage-Sonde: mit echten Zielgrößen und Ankündigung ist das Modell klar besser als die Konstante."""
    Xtr, ytr = G.make_history(1000, 7, nu=0.3)
    Xte, yte = G.make_history(20000, 8, nu=0.3)
    const = M.pinball(yte, np.full(len(yte), np.median(ytr)), 0.5)
    g = M.GBT("quantile", 0.5).fit(Xtr, np.log(ytr))
    assert M.pinball(yte, np.exp(g.predict(Xte)), 0.5) < 0.5 * const
    ll = M.LogLinear().fit(Xtr, ytr)
    assert M.pinball(yte, ll.predict(Xte), 0.5) < 0.5 * const


def test_gbt_against_scikit_learn_when_installed():
    """Gegenprobe der Messreihe (Pinball 0,2 bis 1,1 % daneben bei 64 Klassen): nur wenn scikit-learn da ist; die Demo selbst braucht es nicht."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    Xtr, ytr = G.make_history(1000, 5, nu=0.6)
    Xte, yte = G.make_history(10000, 6, nu=0.6)
    ours = M.GBT("quantile", 0.5, n_rounds=100).fit(Xtr, np.log(ytr))
    theirs = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=5, random_state=0).fit(Xtr, np.log(ytr))
    p_ours = M.pinball(yte, np.exp(ours.predict(Xte)), 0.5)
    p_theirs = M.pinball(yte, np.exp(theirs.predict(Xte)), 0.5)
    assert p_ours == pytest.approx(p_theirs, rel=0.03)
