"""Nachmessung der Messreihe (hafen-planung/messreihe_verweilzeit, ERGEBNIS.md) mit dem Code dieser Demo: 300 Container, Block 6 x 5, Füllgrad 80 %, 200 Blöcke (Seeds 0-199), gepaarte Differenzen.
Toleranzbänder, keine exakten Extreme rauschender Größen (die CI installiert die neueste numpy). Langsam: `pytest -m "not slow"` lässt sie aus."""

from statistics import NormalDist

import numpy as np
import pytest

import vwz_constants as C
import vwz_evaluation as E
import vwz_forecast as F
import vwz_generator as G
import vwz_models as M
import vwz_stk_core as K

pytestmark = pytest.mark.slow
N = 300
NI = 200


def blocks(nu):
    return [E.make_block(6, 5, 80, N, nu, i) for i in range(NI)]


def moves(sample_list, est_fn, place=K.bestfit):
    return np.array([K.run(s.inst, F.est_plain(est_fn(s)), place).moves / N for s in sample_list])


@pytest.fixture(scope="module")
def pure():
    return np.array([E.pure_moves(s) / N for s in blocks(None)])


def learned(nu, rounds=150, alpha=0.7, n_train=1000):
    """Boosting auf log-Verweilzeit mit Quantilverlust (150 Runden wie in der Messreihe), Prognose gegen den Ausgleich."""
    X, y = G.make_history(n_train, 5, nu=nu)
    g = M.GBT("quantile", alpha, n_rounds=rounds).fit(X, np.log(y))
    return moves(blocks(nu), lambda s: s.t_arr + np.maximum(np.exp(g.predict(s.X)), 1e-3))


def diff(v, pure):
    d = v - pure
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def test_reference_and_clairvoyant(pure):
    assert pure.mean() == pytest.approx(0.862, abs=0.004)                                        # Ausgleich 0,862 je Container
    hell = moves(blocks(None), lambda s: s.t_dep)
    d, se = diff(hell, pure)
    assert d == pytest.approx(-0.607, abs=0.01) and se < 0.008


def test_learned_with_announcement_beats_the_reference(pure):
    d6, s6 = diff(learned(0.6), pure)
    d3, s3 = diff(learned(0.3), pure)
    assert d6 == pytest.approx(-0.124, abs=0.015) and d3 == pytest.approx(-0.263, abs=0.015) and d6 < -4 * s6 and d3 < d6 - 0.1          # Messreihe: -0,124 und -0,263


def test_standard_features_alone_do_not_beat_the_reference(pure):
    """Nur Standardmerkmale: gelernt +0,061, das Orakel mit Median +0,043 und mit Quantil 0,7 -0,004 (Messreihe); selbst die exakte bedingte Verteilung schlägt den Ausgleich nicht klar."""
    d, se = diff(learned(None), pure)
    assert d == pytest.approx(0.061, abs=0.015) and d > 3 * se
    z = NormalDist().inv_cdf(0.7)
    d_med, _ = diff(moves(blocks(None), lambda s: s.t_arr + np.exp(s.mu)), pure)
    d_q, se_q = diff(moves(blocks(None), lambda s: s.t_arr + np.exp(s.mu + s.s * z)), pure)
    assert d_med == pytest.approx(0.043, abs=0.015) and abs(d_q) < 3 * se_q + 0.01                # Orakel-Median schlechter, Orakel-Quantil gleichauf


def test_quantile_beats_median_at_fill_80(pure):
    """Quantil 0,7 gegen Median bei ν 0,6 (Boosting 1000): -0,043 je Container (Messreihe), gepaart."""
    X, y = G.make_history(1000, 5, nu=0.6)
    ly = np.log(y)
    g50 = M.GBT("quantile", 0.5, n_rounds=150).fit(X, ly)
    b = blocks(0.6)
    med = moves(b, lambda s: s.t_arr + np.maximum(np.exp(g50.predict(s.X)), 1e-3))
    q70 = learned(0.6)
    d, se = diff(q70, med)
    assert d == pytest.approx(-0.043, abs=0.015) and d < -3 * se


def test_the_gauss_curve_reproduces_n300_sweepA_and_the_tipping_point():
    """n300_sweepA.json: Bestfit mit Gauß-Rauschen sigma 0,1 ... 2,0 auf denselben 200 Blöcken; Kipppunkt sigma 0,78 bei 27 % falsch geordneten Paaren."""
    g = E.gauss_curve((6, 5, 80, 300), n=NI)
    assert g.pure == pytest.approx(0.8617, abs=0.003) and g.hell == pytest.approx(0.255, abs=0.003)
    assert list(g.moves) == pytest.approx([0.403, 0.577, 0.750, 0.852, 0.925, 0.971, 1.014, 1.066], abs=0.003)
    assert list(g.pair_err) == pytest.approx([0.056, 0.129, 0.215, 0.270, 0.307, 0.334, 0.354, 0.383], abs=0.004)
    assert list(g.mae) == pytest.approx([0.080, 0.199, 0.398, 0.596, 0.795, 0.994, 1.193, 1.590], abs=0.01)
    t = E.tipping_point(g)
    assert t.kind == "crosses" and t.sigma == pytest.approx(0.78, abs=0.02) and t.pair_err == pytest.approx(0.27, abs=0.01) and t.mae == pytest.approx(0.62, abs=0.02)


def test_training_size_tips_at_thirty_and_wins_from_three_hundred(pure):
    """ν 0,6, Boosting Quantil 0,7 gegen den Ausgleich: N = 30 +0,103, 300 -0,073, 10 000 -0,159 (Messreihe)."""
    d30, _ = diff(learned(0.6, n_train=30), pure)
    d300, _ = diff(learned(0.6, n_train=300), pure)
    d10k, _ = diff(learned(0.6, n_train=10000), pure)
    assert d30 > 0.05 and d300 < -0.04 and d10k < d300 - 0.03
