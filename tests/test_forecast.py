"""Prognose und Schätzung: Est-Objekt, Rauschen, unsicherheitsbewusste Regel, Lerner-Anbindung, Zwischenspeicher."""

import math

import numpy as np
import pytest

import vwz_constants as C
import vwz_forecast as F
import vwz_generator as G
import vwz_stk_core as K


def test_est_is_a_float_with_extra_information_and_does_not_change_the_rules():
    s = G.generate(6, 5, 0.8, 120, 3)
    e_obj = F.make_est(s.t_arr, s.mu, s.s)
    e_num = F.est_plain([float(e_obj[c]) for c in range(120)])
    assert isinstance(e_obj[5], float) and e_obj[5] == pytest.approx(s.t_arr[5] + math.exp(s.mu[5])) and e_obj[5].t == s.t_arr[5] and e_obj[5].mu == s.mu[5] and e_obj[5].s == s.s[5]
    a = K.run(s.inst, e_obj, K.bestfit, record=True)
    b = K.run(s.inst, e_num, K.bestfit, record=True)
    assert a.steps == b.steps and a.moves == b.moves                                    # Est(float) verhält sich für Bestfit wie die Zahl


@pytest.mark.parametrize("seed", range(6))
def test_clairvoyant_estimate_is_bit_identical_to_the_stapelplanung_with_sigma_zero(seed):
    """Der Hellseher (wahre Abfahrtszeit) trifft in jedem Zustand jedes Ereignisses dieselben Entscheidungen wie run_rule(sigma = 0) mit den wahren Ereignisindizes."""
    s = G.generate(6, 5, 0.8, 150, seed)
    for name, rule in (("bestfit", K.bestfit), ("niedrigster_stapel", K.niedrigster_stapel)):
        a = K.run(s.inst, F.true_est(s), rule, record=True)
        b = K.run_rule(s.inst, name, 0.0, record=True)
        assert a.steps == b.steps and a.cumulative == b.cumulative


def test_noisy_estimate_scales_the_same_random_numbers():
    s = G.generate(6, 5, 0.8, 120, 3)
    e0, e1, e2 = (F.noisy_est(s, sg, 44) for sg in (0.0, 0.5, 1.0))
    assert all(e0[c] == float(s.t_dep[c]) for c in range(120))
    z1 = np.array([e1[c] - s.t_dep[c] for c in range(120)])
    z2 = np.array([e2[c] - s.t_dep[c] for c in range(120)])
    assert np.allclose(z2, 2 * z1) and z1.std() == pytest.approx(0.5 * s.mean_dwell_days, rel=0.25)
    assert F.noisy_est(s, 0.5, 45) != e1 and F.noisy_est(s, 0.5, 44) == e1


def test_the_aware_rule_prefers_the_stack_that_is_least_likely_to_block():
    """Zwei Stapel: oben liegt links ein sicher später abfahrender (P zu blockieren ~ 0), rechts ein sicher früher abfahrender (P ~ 1): die Regel wählt links, egal wo der Index liegt."""
    aware = F.make_aware()

    def est_of(t, mu, s):
        e = F.Est(t + math.exp(mu))
        e.t, e.mu, e.s = t, mu, s
        return e

    own = est_of(0.0, math.log(5.0), 0.1)                                                 # fährt etwa bei 5
    late, early = est_of(0.0, math.log(20.0), 0.1), est_of(0.0, math.log(1.0), 0.1)
    est = {0: early, 1: late}
    assert aware([[0], [1]], 3, own, est) == 1 and aware([[1], [0]], 3, own, {0: early, 1: late}) == 0
    # leerer Stapel: Wahrscheinlichkeit 0; bei Gleichstand zweier sicherer Kandidaten gilt die engste Passung (kleinste oberste Abfahrt), sonst der niedrigste Stapel
    later = est_of(0.0, math.log(40.0), 0.1)
    assert aware([[1], [2], []], 3, own, {1: late, 2: later}) == 0                        # oben 20 statt 40: engere Passung als der leere Stapel
    with pytest.raises(ValueError):
        aware([[0, 0]], 2, own, {0: late})


def test_the_aware_rule_falls_back_to_the_lowest_stack_when_all_are_risky():
    aware = F.make_aware()

    def est_of(t, mu, s):
        e = F.Est(t + math.exp(mu))
        e.t, e.mu, e.s = t, mu, s
        return e

    own = est_of(0.0, math.log(20.0), 0.1)
    early = est_of(0.0, math.log(1.0), 0.1)                                               # fährt sicher vor dem eigenen: jeder Stapel blockiert mit P ~ 1
    est = {i: early for i in range(4)}
    assert aware([[0, 1], [2], [3, 0]], 3, own, est) == 1                                # niedrigster Stapel (wenigste Container)
    assert aware([[0], [1], [2]], 3, own, est) == 0                                      # bei Gleichstand der kleinste Index


def test_the_aware_rule_prefers_the_lower_stack_when_the_risk_is_between_thirty_and_fifty_percent():
    """Bei einem Risiko von rund 40 % gilt der niedrigste Stapel (nicht die engste Passung, die nur unter 0,3 zählt): zwei gleich riskante Stapel, der niedrigere gewinnt gegen den kleineren Index."""
    aware = F.make_aware()

    def est_of(t, mu, s):
        e = F.Est(t + math.exp(mu))
        e.t, e.mu, e.s = t, mu, s
        return e

    own = est_of(0.0, math.log(5.0), 0.5)
    top = lambda: est_of(0.0, math.log(5.675), 0.01)                                            # noqa: E731  fährt mit rund 40 % Wahrscheinlichkeit vor dem eigenen Container
    est = {0: top(), 1: top(), 2: top()}
    assert aware([[0, 1], [2]], 3, own, est) == 1                                             # gleiches Risiko: der niedrigere Stapel, obwohl der Index größer ist


def test_the_boosting_spread_has_a_floor_of_five_percent():
    """Ohne Ankündigung und mit 1000 Trainingscontainern fällt die Streuung eines Containers (Block 268) auf die Untergrenze 0,05, nie darunter."""
    s = G.generate(6, 5, 0.8, 300, 268, nu=None)
    assert F.forecast(C.LEARNER_BOOSTING, None, 1000, 5, s.X).s.min() == pytest.approx(0.05, abs=1e-12)


def test_forecast_of_both_learners_has_the_documented_shape_and_order():
    s = G.generate(6, 5, 0.8, 300, 268, nu=0.6)
    for learner in C.LEARNERS:
        fc = F.forecast(learner, 0.6, 1000, 5, s.X)
        assert isinstance(fc, F.Forecast) and all(len(getattr(fc, f)) == 300 for f in ("median", "mean", "q70", "mu", "s"))
        assert np.all(fc.median > 0) and np.all(fc.s > 0) and np.allclose(np.log(fc.median), fc.mu) and (fc.q70 > fc.median).mean() > 0.9
        assert fc.mean.mean() > fc.median.mean()                                          # lognormal: Mittelwert über dem Median
    with pytest.raises(ValueError):
        F.forecast("unbekannt", 0.6, 1000, 5, s.X)
    with pytest.raises(ValueError):
        F.forecast_curve("unbekannt", 0.6, 1000, 5, s.X)


def test_linear_forecast_uses_a_constant_log_spread_and_boosting_a_per_container_spread():
    s = G.generate(6, 5, 0.8, 300, 268, nu=0.6)
    lin = F.forecast(C.LEARNER_LINEAR, 0.6, 1000, 5, s.X)
    boo = F.forecast(C.LEARNER_BOOSTING, 0.6, 1000, 5, s.X)
    assert np.all(lin.s == lin.s[0]) and lin.s[0] == pytest.approx(0.414321065587581, rel=1e-9)          # Messreihe: LogLinear auf history(1000, Seed 5, nu 0,6)
    assert boo.s.min() >= C.S_FLOOR and boo.s.std() > 0.05                                # heteroskedastisch: Export streut anders als Leer
    art = s.X[:, 0]
    assert boo.s[art == 3].mean() > boo.s[art == 1].mean()
    assert np.allclose(lin.q70, np.exp(lin.mu + lin.s * 0.5244005101), rtol=1e-6) and np.allclose(lin.mean, np.exp(lin.mu + lin.s ** 2 / 2))


def test_boosting_q70_is_bit_identical_to_the_messreihe():
    """Referenzwert aus hafen-planung/messreihe_verweilzeit (models.GBT, 100 Runden, Historie 1000 / Seed 5 / nu 0,6, Block Seed 268)."""
    s = G.generate(6, 5, 0.8, 300, 268, nu=0.6)
    pred = F.forecast_curve(C.LEARNER_BOOSTING, 0.6, 1000, 5, s.X)
    assert pred.sum() == pytest.approx(1840.0910541485457, rel=1e-9) and pred[:3] == pytest.approx([5.959155564465083, 6.392485952060709, 4.387040798539201], rel=1e-9)
    assert F.forecast(C.LEARNER_BOOSTING, 0.6, 1000, 5, s.X).q70 == pytest.approx(pred)


def test_curve_forecast_equals_the_q70_of_the_full_forecast_and_the_group_mean_ignores_the_announcement():
    s = G.generate(6, 5, 0.8, 300, 4, nu=0.6)
    for learner in C.LEARNERS:
        assert np.allclose(F.forecast_curve(learner, 0.6, 300, 2, s.X), F.forecast(learner, 0.6, 300, 2, s.X).q70)
    g1 = F.forecast_curve(C.LEARNER_GROUP, 0.6, 300, 2, s.X)
    g2 = F.forecast_curve(C.LEARNER_GROUP, 0.15, 300, 2, s.X)
    g3 = F.forecast_curve(C.LEARNER_GROUP, None, 300, 2, s.X)
    assert np.array_equal(g1, g2) and np.array_equal(g1, g3) and len(set(g1.round(9))) == 4        # ein Wert je Containerart, unabhängig von der Ankündigung


def test_models_are_trained_once_per_setting_and_are_deterministic():
    F._boost.cache_clear()
    a = F._boost(0.6, 100, 9, "q70")
    assert F._boost(0.6, 100, 9, "q70") is a and F._boost(0.6, 100, 9, "q50") is not a and F._boost(0.6, 100, 10, "q70") is not a
    X, y = F.history(0.6, 100, 9)
    assert F.history(0.6, 100, 9)[0] is X and np.array_equal(y, G.make_history(100, 9, nu=0.6)[1])
    assert F._linear(0.6, 100, 9) is F._linear(0.6, 100, 9)
    with pytest.raises(Exception):
        F._boost(0.6, 100, 9, "q999x")


def test_more_training_data_makes_the_signal_sharper():
    s = G.generate(6, 5, 0.8, 300, 4, nu=0.3)
    err = {n: np.mean(np.abs(np.log(F.forecast(C.LEARNER_LINEAR, 0.3, n, 1, s.X).median) - np.log(s.dwell))) for n in (30, 3000)}
    assert err[3000] < err[30]
