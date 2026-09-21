"""Auswertung: Paarfehler gegen eine Doppelschleife, Regeln auf dem Block, Bezug gegen unabhängige Nachrechnung, Hellseher bitgleich, Stichprobe, Urteil an der Schwelle, Verteilung, Gauß-Kurve,
Kipppunkt, äquivalentes Rauschen, Lernkurve, Meldung, Hilfen der Blockansicht."""

from types import SimpleNamespace

import numpy as np
import pytest

import vwz_constants as C
import vwz_evaluation as E
import vwz_forecast as F
import vwz_generator as G
import vwz_stk_core as K
from helpers import naive_pair_error, reference_pure_moves

LOW, MED, Q70, AWARE, MEAN, SAME, HELL = C.RULE_LOWEST, C.RULE_MEDIAN, C.RULE_Q70, C.RULE_AWARE, C.RULE_MEAN, C.RULE_LOWEST_SAME, C.RULE_HELL
P = E.params_from_preset(C.PRESETS["Mit Ankündigung"])


def row(rein, gel, hell=0.3, seed=0):
    return E.Row(seed, {LOW: rein, Q70: gel, HELL: hell, MED: gel, MEAN: gel, AWARE: gel, SAME: gel})


def rows_from_diffs(diffs, rein=1.0):
    return tuple(row(rein, rein + d, seed=i) for i, d in enumerate(diffs))


# ---------------------------------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------------------------------
def test_params_properties_and_key():
    p = E.Params(6, 5, 80, 300, 2, 3, "boosting", 5)
    assert p.nu == 0.6 and p.n_train == 1000 and p.fill == 0.8 and p.block_key == (6, 5, 80, 300) and p.key == (6, 5, 80, 300, 2, 3, "boosting", 5)
    assert E.Params(6, 5, 80, 300, 0, 0, "linear", 0).nu is None and E.Params(6, 5, 80, 300, 4, 5, "linear", 0).n_train == 10000
    assert E.params_from_preset(C.PRESETS["Voller Block"]) == E.Params(6, 5, 100, 300, 2, 3, "boosting", 5)
    assert E.block_of(p, 268) is E.block_of(p, 268)                                       # im Prozess gemerkt


# ---------------------------------------------------------------------------------------------------
# Paarfehler und Güte
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_pair_counts_against_a_double_loop(seed):
    s = G.generate(6, 5, 0.8, 60, seed)
    rng = np.random.default_rng(seed)
    for est in (s.t_dep, s.t_arr + s.dwell * rng.lognormal(0, 0.5, 60), rng.random(60) * 30, np.round(s.t_arr + s.dwell, 0)):        # exakt, verrauscht, zufällig, mit Gleichständen
        assert E.pair_counts(s.t_arr, s.t_dep, est) == naive_pair_error(s.t_arr, s.t_dep, est)


def test_pair_counts_hand_case():
    t_arr = np.array([0.0, 1.0, 5.0])
    t_dep = np.array([4.0, 3.0, 9.0])                     # 0 und 1 gleichzeitig anwesend, 2 kommt erst nach ihnen
    assert E.pair_counts(t_arr, t_dep, np.array([4.0, 3.0, 9.0])) == (0, 1)
    assert E.pair_counts(t_arr, t_dep, np.array([3.0, 4.0, 9.0])) == (1, 1)               # falsch herum
    assert E.pair_counts(t_arr, t_dep, np.array([3.0, 3.0, 9.0])) == (1, 1)               # Gleichstand der Schätzung zählt als falsch
    assert E.pair_counts(np.array([0.0, 4.0]), np.array([4.0, 6.0]), np.array([9.0, 1.0])) == (0, 0)      # Aufenthalte berühren sich nur: kein Paar (strikt)


def test_quality_hand_case_and_pooling():
    a = SimpleNamespace(t_arr=np.array([0.0, 1.0]), t_dep=np.array([4.0, 3.0]), dwell=np.array([4.0, 2.0]), mean_dwell_days=3.0)
    b = SimpleNamespace(t_arr=np.array([0.0, 1.0]), t_dep=np.array([5.0, 6.0]), dwell=np.array([5.0, 5.0]), mean_dwell_days=5.0)
    q = E.quality_est([a], [np.array([3.0, 5.0])])                                        # Fehler 1 und 2, mittlere Verweilzeit 3; Reihenfolge falsch herum
    assert q.mae == pytest.approx(1.5 / 3.0) and q.pair_err == 1.0
    q2 = E.quality_est([a, b], [np.array([4.0, 3.0]), np.array([5.0, 6.0])])              # perfekt
    assert q2.mae == 0.0 and q2.pair_err == 0.0
    q3 = E.quality_of([a, b], [np.array([3.0, 3.0]), np.array([5.0, 5.0])])               # a: Prognose 3 und 3 Tage (Schätzung 3 und 4: falsch herum); b perfekt; d0 = Mittel der Blockmittel = 4
    assert q3.mae == pytest.approx(np.mean([1.0, 1.0, 0.0, 0.0]) / 4.0) and q3.pair_err == pytest.approx(0.5)
    assert E.quality_est([a], [np.array([3.0, 5.0])], d0=1.5).mae == pytest.approx(1.0)
    assert E.mae_days(np.array([1.0, 5.0]), np.array([2.0, 3.0])) == pytest.approx(1.5)


# ---------------------------------------------------------------------------------------------------
# Regeln auf einem Block
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_the_reference_is_the_lowest_stack_without_any_forecast(seed):
    s = G.generate(6, 5, 0.8, 100, seed)
    assert E.pure_moves(s) == reference_pure_moves(s.inst)
    assert E.rule_moves(s, LOW, None).moves == E.pure_moves(s)


@pytest.mark.parametrize("seed", range(6))
def test_clairvoyant_rule_is_bit_identical_to_the_stapelplanung(seed):
    s = G.generate(6, 5, 0.8, 150, seed)
    a = E.rule_moves(s, HELL, None, record=True)
    b = K.run_rule(s.inst, "bestfit", 0.0, record=True)
    assert a.steps == b.steps and a.moves == b.moves


@pytest.mark.parametrize("seed", range(4))
def test_an_exact_announcement_gives_the_clairvoyant_result(seed):
    """Grenzfall nu = 0: die Ankündigung ist die wahre Verweilzeit, Ankunft + Ankündigung die wahre Abfahrt; Bestfit darauf ist der Hellseher, Zustand für Zustand."""
    s = G.generate(6, 5, 0.8, 100, seed, nu=0.0)
    assert np.array_equal(s.t_arr + s.X[:, C.COL_ANNOUNCED], s.t_dep)
    a = K.run(s.inst, F.est_plain(s.t_arr + s.X[:, C.COL_ANNOUNCED]), K.bestfit, record=True)
    assert a.steps == E.rule_moves(s, HELL, None, record=True).steps


def test_rules_use_the_forecast_they_are_named_after():
    s = G.generate(6, 5, 0.8, 150, 2, nu=0.6)
    fc = F.forecast(C.LEARNER_LINEAR, 0.6, 300, 5, s.X)
    for key, est in ((MED, fc.median), (Q70, fc.q70), (MEAN, fc.mean)):
        want = K.run(s.inst, F.est_plain(s.t_arr + est), K.bestfit)
        assert E.rule_moves(s, key, fc).moves == want.moves
    same = K.run(s.inst, F.est_plain(s.t_arr + fc.q70), K.niedrigster_stapel)             # Einlagern nach niedrigstem Stapel, Umstapeln mit der Prognose (Bestfit)
    assert E.rule_moves(s, SAME, fc).moves == same.moves
    aware = K.run(s.inst, F.make_est(s.t_arr, fc.mu, fc.s), F.make_aware())
    assert E.rule_moves(s, AWARE, fc).moves == aware.moves
    assert E.rule_moves(s, Q70, fc, record=True).steps and not E.rule_moves(s, Q70, fc).steps
    assert E._clip(np.array([-1.0, 0.0, 2.0])).tolist() == [1e-3, 1e-3, 2.0]


def test_forecast_of_maps_each_rule_to_its_forecast():
    fc = F.Forecast(*(np.full(3, v) for v in (1.0, 2.0, 3.0, 0.5, 0.2)))
    assert [E.forecast_of(k, fc)[0] for k in (MED, MEAN, Q70, AWARE, SAME)] == [1.0, 2.0, 3.0, 1.0, 3.0]
    assert E.forecast_of(LOW, fc) is None and E.forecast_of(HELL, fc) is None


def test_run_rules_returns_the_seven_rules_with_their_quality():
    outs = E.run_rules(P, 268)
    assert [o.key for o in outs] == list(C.RULE_KEYS) and all(o.label == C.RULE_LABELS[o.key] and o.result.steps for o in outs)
    by = {o.key: o for o in outs}
    assert by[LOW].quality is None and by[LOW].est_days is None and by[HELL].quality == E.Quality(0.0, 0.0) and by[HELL].est_days is None
    assert by[AWARE].quality == by[MED].quality and np.array_equal(by[AWARE].est_days, by[MED].est_days)        # dieselbe Prognose (Median)
    assert by[SAME].quality == by[Q70].quality and by[MEAN].quality.mae != by[Q70].quality.mae
    assert by[Q70].quality.mae > by[MED].quality.mae                                        # das Quantil hat den größeren mittleren Fehler
    assert by[Q70].quality.pair_err < by[MED].quality.pair_err                              # aber weniger falsch geordnete Paare
    assert E.outcome_of(outs, Q70) is by[Q70] and by[Q70].moves == by[Q70].result.moves
    assert all(np.all(o.est_days > 0) for o in outs if o.est_days is not None)
    assert not E.run_rules(P, 268, record=False)[0].result.steps


def test_preset_blocks_at_seed_268_reproduce_the_messreihe():
    """Ausgleich / gelernt (Boosting, Quantil 0,7, 100 Runden, Trainings-Seed 5) / Hellseher am gezeigten Block: presets_final300.json der Messreihe."""
    expected = {"Nur Standard": (238, 252, 79), "Mit Ankündigung": (238, 222, 79), "Zu wenig Daten": (238, 281, 79), "Sehr verlässlich": (238, 160, 79), "Voller Block": (355, 310, 148)}
    for name, (rein, gel, hell) in expected.items():
        outs = E.run_rules(E.params_from_preset(C.PRESETS[name]), 268, record=False)
        assert (E.outcome_of(outs, LOW).moves, E.outcome_of(outs, Q70).moves, E.outcome_of(outs, HELL).moves) == (rein, gel, hell), name


def test_other_rules_at_the_default_block_are_pinned_as_regression_values():
    """Regressionswerte dieser Demo (keine Messreihe): Median, Unsicherheitsbewusst, Mittelwert, Ausgleich mit Umstapeln nach Prognose am Block 268, Standard-Preset."""
    outs = {o.key: o.moves for o in E.run_rules(P, 268, record=False)}
    assert outs == {LOW: 238, MED: 230, Q70: 222, AWARE: 242, MEAN: 215, SAME: 263, HELL: 79}


# ---------------------------------------------------------------------------------------------------
# Stichprobe
# ---------------------------------------------------------------------------------------------------
def test_sample_structure_and_reproduction_of_the_preset_sample():
    sm = E.sample(P)
    assert len(sm) == 60 and [r.seed for r in sm.rows] == list(range(60)) and sm.n_containers == 300 and set(sm.rows[0].m) == set(C.RULE_KEYS)
    assert E.mean_of(sm.rows, LOW) == pytest.approx(0.8577777777777776, abs=0.002) and E.mean_of(sm.rows, Q70) == pytest.approx(0.7372222222222222, abs=0.003)
    assert E.mean_of(sm.rows, HELL) == pytest.approx(0.25027777777777777, abs=0.002)
    d, se = E.paired(sm.rows, Q70)
    assert d == pytest.approx(-0.12055555555555553, abs=0.003) and se == pytest.approx(0.01085028130098594, abs=0.002)
    q = sm.quality
    assert q[LOW] is None and q[HELL] == E.Quality(0.0, 0.0) and q[Q70].pair_err == pytest.approx(0.213899, abs=0.005) and q[Q70].mae == pytest.approx(0.338807, abs=0.005)
    assert q[AWARE] == q[MED] and q[SAME] == q[Q70]
    assert sm.rows[3].m[LOW] == E.pure_moves(E.block_of(P, 3)) / 300 and E.values(sm.rows, LOW) == [r.m[LOW] for r in sm.rows]


def test_sample_rows_are_moves_per_container_of_the_block_size():
    p = E.Params(6, 5, 80, 100, 2, 2, "linear", 5)
    sm = E.sample(p, n=2, rules=(LOW, HELL))
    for i, r in enumerate(sm.rows):
        s = E.block_of(p, i)
        assert r.m[LOW] == E.pure_moves(s) / 100 and r.m[HELL] == K.run_rule(s.inst, "bestfit", 0.0).moves / 100 and sm.n_containers == 100


def test_sample_light_path_equals_the_full_path_for_the_quantile_rule():
    full = E.sample(P, n=6)
    light = E.sample(P, n=6, rules=(LOW, Q70, HELL))
    assert set(light.rows[0].m) == {LOW, Q70, HELL} and set(light.quality) == {LOW, Q70, HELL}
    for a, b in zip(full.rows, light.rows):
        assert (a.m[LOW], a.m[Q70], a.m[HELL]) == (b.m[LOW], b.m[Q70], b.m[HELL])
    assert full.quality[Q70] == light.quality[Q70] and E.sample(P, n=3, seed0=7).rows[0].seed == 7


def test_sample_is_deterministic_and_independent_of_the_shown_seed():
    a, b = E.sample(P, n=4), E.sample(P, n=4)
    assert a.rows == b.rows and a.quality == b.quality


def test_sample_mean_and_paired_on_hand_values():
    rows = rows_from_diffs([-0.2, -0.4], rein=1.0)
    assert E.mean_of(rows, Q70) == pytest.approx(0.7) and E.diffs(rows, Q70) == pytest.approx([-0.2, -0.4]) and E.diffs(rows, LOW) == [0.0, 0.0]
    d, se = E.paired(rows, Q70)
    assert d == pytest.approx(-0.3) and se == pytest.approx(0.1)                             # stdev 0,1414 / sqrt 2
    assert E.paired(rows[:1], Q70) == (pytest.approx(-0.2), 0.0)                             # ein Block: kein Standardfehler


# ---------------------------------------------------------------------------------------------------
# Urteil und Verteilung
# ---------------------------------------------------------------------------------------------------
def spread_rows(m, s=0.3, n=10):
    """n Blöcke mit gepaarter Differenz m +- s (je zur Hälfte): Mittel m, Standardfehler s / 3."""
    return rows_from_diffs([m + s] * (n // 2) + [m - s] * (n // 2))


def test_verdict_threshold_is_two_standard_errors():
    assert E.paired(spread_rows(-0.19), Q70) == pytest.approx((-0.19, 0.1))
    assert E.verdict(spread_rows(-0.19), Q70).kind == "unclear" and E.verdict(spread_rows(-0.21), Q70).kind == "better"
    assert E.verdict(spread_rows(0.19), Q70).kind == "unclear" and E.verdict(spread_rows(0.21), Q70).kind == "worse"
    v = E.verdict(spread_rows(-0.5), Q70)
    assert v.kind == "better" and v.diff == pytest.approx(-0.5) and v.se == pytest.approx(0.1) and v.pct == pytest.approx(-50.0) and v.n == 10
    assert E.verdict(spread_rows(0.5), Q70).pct == pytest.approx(50.0)
    assert E.verdict(rows_from_diffs([0.0, 0.0, 0.0]), Q70).kind == "unclear"              # kein Unterschied: kein Urteil
    assert E.verdict(rows_from_diffs([-0.1] * 5, rein=0.0), Q70).pct is None                # Bezug 0: kein Prozentwert


def test_distribution_hand_values():
    rows = rows_from_diffs([-0.3, -0.1, 0.0, 0.0, 0.2, -0.2, -0.4, 0.1, 0.0, -0.05])
    d = E.distribution(rows, Q70)
    assert d.better == pytest.approx(0.5) and d.worse == pytest.approx(0.2) and d.equal == pytest.approx(0.3) and d.better + d.equal + d.worse == pytest.approx(1.0)
    assert d.median == pytest.approx(-0.025)
    assert d.top_decile_share == pytest.approx(0.4 / 0.75)                                    # bester Block (-0,4) von netto 0,75 Ersparnis (1,05 gespart, 0,3 draufgelegt)
    none = E.distribution(rows_from_diffs([0.1, 0.2, 0.3]), Q70)
    assert none.top_decile_share is None and none.better == 0.0 and none.worse == 1.0
    top20 = E.distribution(rows_from_diffs([-0.1] * 15 + [-0.5] * 5), Q70)                    # ein Zehntel von 20 Blöcken sind 2
    assert top20.top_decile_share == pytest.approx(1.0 / 4.0)                                 # 2 mal 0,5 von 4,0 Ersparnis
    assert E.distribution(rows_from_diffs([-0.1]), Q70).top_decile_share == 1.0             # ein Block: mindestens ein Block zählt
    top15 = E.distribution(rows_from_diffs([-0.5] * 3 + [-0.1] * 12), Q70)                    # ein Zehntel von 15 Blöcken, aufgerundet: 2
    assert top15.top_decile_share == pytest.approx(1.0 / 2.7)                                 # 2 mal 0,5 von 2,7 Ersparnis


# ---------------------------------------------------------------------------------------------------
# Gauß-Kurve, Kipppunkt, äquivalentes Rauschen
# ---------------------------------------------------------------------------------------------------
def test_gauss_curve_is_monotone_and_bracketed_by_reference_and_clairvoyant():
    g = E.gauss_curve((6, 5, 80, 300), n=12)
    assert g.sigmas == C.GAUSS_SIGMAS and g.n_instances == 12 and len(g.moves) == len(g.mae) == len(g.pair_err) == 8
    assert all(a < b for a, b in zip(g.moves, g.moves[1:])) and all(a < b for a, b in zip(g.pair_err, g.pair_err[1:])) and all(a < b for a, b in zip(g.mae, g.mae[1:]))
    assert g.hell < g.moves[0] < g.pure < g.moves[-1] and 0 < g.pair_err[0] < g.pair_err[-1] < 0.5
    assert g.mae[3] == pytest.approx(0.75 * 0.8 / 0.8 * 0.8, abs=0.3)                          # grobe Größenordnung: MAE ~ 0,8 sigma bei Gauß
    assert E.gauss_curve((6, 5, 80, 300), n=12) == g                                          # deterministisch


def test_gauss_curve_mae_is_about_point_eight_sigma():
    g = E.gauss_curve((4, 4, 60, 100), n=10)
    for s, m in zip(g.sigmas, g.mae):
        assert m == pytest.approx(0.7979 * s, rel=0.12)                                        # E|N(0, s)| = 0,798 s in mittleren Standzeiten


def test_tipping_point_interpolation_and_edge_cases():
    curve = lambda moves, pure: E.GaussCurve((0.5, 1.0, 1.5), moves, (0.4, 0.8, 1.2), (0.2, 0.3, 0.4), pure, 0.1, 10)          # noqa: E731
    t = E.tipping_point(curve((0.5, 0.9, 1.1), 1.0))                                           # von -0,1 nach +0,1 zwischen 1,0 und 1,5 ... zuerst -0,1 (sigma 1,0), dann +0,1
    assert t.kind == "crosses" and t.sigma == pytest.approx(1.25) and t.pair_err == pytest.approx(0.35) and t.mae == pytest.approx(1.0)
    t = E.tipping_point(curve((0.5, 1.0, 1.4), 1.0))                                           # 0,0 gilt als nicht besser: Wechsel bei sigma 1,0
    assert t.kind == "crosses" and t.sigma == pytest.approx(1.0) and t.pair_err == pytest.approx(0.3)
    assert E.tipping_point(curve((0.5, 0.6, 0.7), 1.0)).kind == "always_better"
    assert E.tipping_point(curve((1.0, 1.2, 1.3), 1.0)).kind == "always_worse" and E.tipping_point(curve((1.1, 1.2, 1.3), 1.0)).kind == "always_worse"
    t = E.tipping_point(curve((0.9, 0.99, 1.2), 1.0))
    assert t.sigma == pytest.approx(1.0 + 0.5 * 0.01 / 0.21)


def test_equivalent_sigma_reads_the_curve_at_the_moves():
    g = E.GaussCurve((0.5, 1.0, 2.0), (0.4, 0.8, 1.2), (0.4, 0.8, 1.6), (0.2, 0.3, 0.4), 0.9, 0.1, 10)
    assert E.equivalent_sigma(g, 0.6) == (pytest.approx(0.75), "inside") and E.equivalent_sigma(g, 1.0) == (pytest.approx(1.5), "inside")
    assert E.equivalent_sigma(g, 0.4) == (0.5, "inside") and E.equivalent_sigma(g, 1.2) == (2.0, "inside")
    assert E.equivalent_sigma(g, 0.3) == (0.5, "below") and E.equivalent_sigma(g, 1.3) == (2.0, "above")
    flat = E.GaussCurve((0.5, 1.0), (0.4, 0.4), (0.4, 0.8), (0.2, 0.3), 0.9, 0.1, 10)
    assert E.equivalent_sigma(flat, 0.4) == (0.5, "inside")


# ---------------------------------------------------------------------------------------------------
# Lernkurve
# ---------------------------------------------------------------------------------------------------
def test_learning_curve_structure_and_the_story_of_the_learners():
    p = E.Params(6, 5, 80, 300, 3, 3, "linear", 5)                                             # gute Ankündigung, lineare Regression
    lc = E.learning_curve(p, n=10)
    assert lc.sizes == C.TRAIN_SIZES and lc.n_instances == 10 and lc.seeds == (5, 6, 7, 8)
    assert set(lc.series) == set(lc.band) == {"linear", "boosting", "gruppe"} and len(lc.pair_err) == len(lc.moves) == 6
    for k in lc.series:
        assert len(lc.series[k]) == len(lc.band[k]) == 6
        for v, (lo, hi) in zip(lc.series[k], lc.band[k]):
            assert lo <= v <= hi                                                               # das Band enthält den Wert des eingestellten Seeds
    assert all(lc.band[k][i][0] == lc.band[k][i][1] == lc.series[k][i] for k in lc.band for i in range(3, 6))     # ab 1000 Containern nur ein Seed
    assert any(lc.band[k][0][0] < lc.band[k][0][1] for k in lc.band)                           # bei 30 Containern streuen die Seeds
    assert all(v > 0 for v in lc.series["gruppe"]) and lc.series["linear"][-1] < -0.15 and lc.series["boosting"][-1] < -0.1     # Gruppenmittel nie besser; mit Ankündigung gewinnt das Lernen
    assert lc.series["linear"][0] > lc.series["linear"][3] and lc.pair_err[0] > lc.pair_err[-1]      # mehr Daten, weniger falsch geordnete Paare
    assert lc.moves[3] == pytest.approx(E.mean_of(E.sample(p, n=10).rows, Q70), abs=1e-9)      # der Punkt der eingestellten Menge ist die Stichprobe des Kernabschnitts


def test_learning_curve_selected_learner_defines_pair_error_and_moves():
    a = E.learning_curve(E.Params(6, 5, 80, 300, 3, 3, "linear", 5), n=6)
    b = E.learning_curve(E.Params(6, 5, 80, 300, 3, 3, "boosting", 5), n=6)
    assert a.series["linear"] == b.series["linear"] and a.moves != b.moves and a.pair_err != b.pair_err
    assert a.moves == pytest.approx(tuple(a.series["linear"][i] + E.mean_of(E.sample(E.Params(6, 5, 80, 300, 3, 3, "linear", 5), n=6).rows, LOW) for i in range(6)))


def test_learning_curve_seeds_wrap_around_the_seed_range():
    lc = E.learning_curve(E.Params(6, 5, 80, 100, 3, 0, "linear", 98), n=3)
    assert lc.seeds == (98, 99, 0, 1)


# ---------------------------------------------------------------------------------------------------
# Meldung
# ---------------------------------------------------------------------------------------------------
def outcomes_with(rein, gel):
    r = lambda m: K.Result(m, 300, 0, 0, (), ())                                               # noqa: E731
    o = lambda key, m: E.Outcome(key, C.RULE_LABELS[key], r(m), None, None)                    # noqa: E731
    return (o(LOW, rein), o(MED, gel), o(Q70, gel), o(AWARE, gel))


def test_diagnosis_kinds_and_flags():
    out = outcomes_with(238, 222)
    p = lambda a, t: E.Params(6, 5, 80, 300, a, t, "boosting", 5)                              # noqa: E731
    d = E.diagnose(Q70, spread_rows(-0.5), out, p(2, 3))
    assert d.kind == "helps" and not d.standard_only and not d.few_data and (d.block_rule, d.block_ref) == (222, 238) and d.verdict.kind == "better"
    assert E.diagnose(Q70, spread_rows(0.5), out, p(2, 3)).kind == "tipped" and E.diagnose(Q70, spread_rows(0.05), out, p(2, 3)).kind == "unclear"
    assert E.diagnose(Q70, spread_rows(-0.5), out, p(0, 3)).standard_only and not E.diagnose(Q70, spread_rows(-0.5), out, p(1, 3)).standard_only
    assert E.diagnose(Q70, spread_rows(-0.5), out, p(2, 2)).few_data is False and E.diagnose(Q70, spread_rows(-0.5), out, p(2, 1)).few_data is True      # 300 Container: genug, 100: zu wenig


def test_diagnosis_text_for_every_kind_and_both_additions():
    out = outcomes_with(238, 222)
    p = E.Params(6, 5, 80, 300, 2, 3, "boosting", 5)
    helps = E.diagnosis_text(E.diagnose(Q70, spread_rows(-0.5), out, p), Q70, 268)
    assert helps.startswith("Die Prognose lohnt sich: Bestfit Quantil 0,7 braucht im Mittel über 10 Blöcke 50 % weniger Umstapelungen als der Ausgleich (-0.500 je Container, Standardfehler 0.100).")
    assert "Am gezeigten Block (Seed 268): 222 gegen 238 Umstapelungen" in helps and "Nur Standardmerkmale" not in helps and "Trainingscontainer" not in helps
    tipped = E.diagnosis_text(E.diagnose(MED, spread_rows(0.5), out, p), MED, 1)
    assert tipped.startswith("Die Prognose kippt: Bestfit Median braucht im Mittel über 10 Blöcke 50 % mehr Umstapelungen als der Ausgleich (+0.500 je Container, Standardfehler 0.100).") and "Ausgleich ist besser" in tipped
    unclear = E.diagnosis_text(E.diagnose(AWARE, spread_rows(0.05), out, p), AWARE, 1)
    assert unclear.startswith("Kein klarer Unterschied: Unsicherheitsbewusst und der Ausgleich liegen im Mittel über 10 Blöcke innerhalb des Rauschens (+0.050 je Container, Standardfehler 0.100).")
    std = E.diagnosis_text(E.diagnose(Q70, spread_rows(0.5), out, E.Params(6, 5, 80, 300, 0, 3, "boosting", 5)), Q70, 1)
    assert "Nur Standardmerkmale (Art, Kunde, Zoll, Wochenende, Vorlauf)" in std and "+0,043 mit Median, -0,004 mit Quantil 0,7" in std and "Erst eine verlässliche Ankündigung trägt." in std
    few = E.diagnosis_text(E.diagnose(Q70, spread_rows(0.5), out, E.Params(6, 5, 80, 300, 2, 0, "boosting", 5)), Q70, 1)
    assert "Weniger als 300 Trainingscontainer" in few and "Trainings-Seed ändern" in few and "Nur Standardmerkmale" not in few
    both = E.diagnosis_text(E.diagnose(Q70, spread_rows(0.5), out, E.Params(6, 5, 80, 300, 0, 0, "boosting", 5)), Q70, 1)
    assert "Nur Standardmerkmale" in both and "Weniger als 300 Trainingscontainer" in both
    pct_none = E.diagnosis_text(E.diagnose(Q70, rows_from_diffs([-0.1] * 5, rein=0.0), out, p), Q70, 1)
    assert "0.100 weniger Umstapelungen" in pct_none                                          # Bezug 0: kein Prozentwert, Betrag stattdessen
    pct_none = E.diagnosis_text(E.diagnose(Q70, rows_from_diffs([0.1, 0.11, 0.12, 0.1, 0.1], rein=0.0), out, p), Q70, 1)
    assert "0.106 mehr Umstapelungen" in pct_none


# ---------------------------------------------------------------------------------------------------
# Hilfen der Blockansicht
# ---------------------------------------------------------------------------------------------------
def test_state_at_suggested_event_and_describe_step():
    outs = E.run_rules(P, 268)
    base = E.outcome_of(outs, LOW)
    inst = E.block_of(P, 268).inst
    stacks, moved, arrived, step = E.state_at(inst, base.result, 0)
    assert stacks == ((),) * 6 and moved == () and arrived is None and step is None
    k = E.suggested_event(base.result)
    st_k = base.result.steps[k - 1]
    assert st_k.kind == "D" and st_k.moved and E.state_at(inst, base.result, k)[3] is st_k and E.state_at(inst, base.result, k)[1] == st_k.moved
    occ = lambda i: sum(len(s) for s in base.result.steps[i - 1].stacks)                        # noqa: E731
    assert all(occ(k) >= occ(i) for i, s in enumerate(base.result.steps, 1) if s.kind == "D" and s.moved)      # höchste Belegung unter allen Abholungen mit Umstapeln
    a = next(i for i, s in enumerate(base.result.steps, 1) if s.kind == "A")
    assert E.state_at(inst, base.result, a)[2] == base.result.steps[a - 1].container
    assert E.describe_step(None, 0, 600) == "Ereignis 0 von 600: Der Block ist noch leer."
    A = K.Step("A", 4, 2, (), ((), (), (4,)), 0)
    assert E.describe_step(A, 3, 10) == "Ereignis 3 von 10: Container 4 kommt an und wird auf Stapel 3 gelegt."
    D0 = K.Step("D", 5, 0, (), ((), (), ()), 0)
    assert E.describe_step(D0, 4, 10) == "Ereignis 4 von 10: Container 5 wird aus Stapel 1 abgeholt, ohne Umstapelung."
    D1 = K.Step("D", 5, 0, (7,), ((), (), ()), 1)
    assert E.describe_step(D1, 4, 10) == "Ereignis 4 von 10: Container 5 wird aus Stapel 1 abgeholt, dafür muss 1 Container umgestapelt werden."
    D2 = K.Step("D", 5, 1, (7, 8), ((), (), ()), 2)
    assert E.describe_step(D2, 4, 10) == "Ereignis 4 von 10: Container 5 wird aus Stapel 2 abgeholt, dafür müssen 2 Container umgestapelt werden."


def test_suggested_event_takes_the_first_of_equally_full_relocations():
    two = K.Result(2, 2, 2, 1, (0, 1, 2), (K.Step("D", 0, 0, (5,), ((1,), (2,), ()), 1), K.Step("D", 1, 0, (6,), ((3,), (4,), ()), 2)))
    assert E.suggested_event(two) == 1                                                       # gleiche Belegung: die frühere Abholung
    fuller = K.Result(2, 2, 2, 1, (0, 1, 2), (K.Step("D", 0, 0, (5,), ((1,), (), ()), 1), K.Step("D", 1, 0, (6,), ((3,), (4,), ()), 2)))
    assert E.suggested_event(fuller) == 2                                                    # höhere Belegung gewinnt


def test_suggested_event_without_any_relocation_takes_the_first_fullest_moment():
    tie = K.Result(0, 2, 0, 1, (0, 0, 0, 0), (K.Step("A", 0, 0, (), ((0,), (), ()), 0), K.Step("A", 1, 1, (), ((0,), (1,), ()), 0), K.Step("D", 0, 0, (), ((), (1,), ()), 0),
                                              K.Step("A", 2, 0, (), ((2,), (1,), ()), 0)))
    assert E.suggested_event(tie) == 2                                                       # zwei Container zuerst nach Ereignis 2 (später wieder): der frühere zählt


def test_gauss_curve_reproduces_the_60_block_sample_of_the_messreihe():
    """60 Blöcke (Seeds 0-59), Block 6 x 5, 80 %, 300 Container: dieselben Blöcke wie n300_sweepA.json, die ersten 60 (gemessen mit diesem Code)."""
    g = E.gauss_curve((6, 5, 80, 300))
    assert list(g.moves) == pytest.approx([0.396, 0.576, 0.751, 0.843, 0.927, 0.969, 1.010, 1.061], abs=0.003)
    assert g.pure == pytest.approx(0.8578, abs=0.002) and g.hell == pytest.approx(0.2503, abs=0.002)
    assert list(g.pair_err) == pytest.approx([0.056, 0.129, 0.215, 0.269, 0.306, 0.333, 0.354, 0.382], abs=0.005)
    assert list(g.mae) == pytest.approx([0.080, 0.199, 0.399, 0.598, 0.798, 0.997, 1.196, 1.595], abs=0.01)
    t = E.tipping_point(g)
    assert t.kind == "crosses" and t.sigma == pytest.approx(0.793, abs=0.02) and t.pair_err == pytest.approx(0.276, abs=0.01)


def test_suggested_event_without_any_relocation_falls_back_to_the_fullest_moment():
    quiet = K.Result(0, 3, 0, 1, (0, 0, 0, 0, 0, 0), (K.Step("A", 0, 0, (), (((0,)), (), ()), 0), K.Step("A", 1, 1, (), ((0,), (1,), ()), 0), K.Step("A", 2, 2, (), ((0,), (1,), (2,)), 0),
                                                       K.Step("D", 0, 0, (), ((), (1,), (2,)), 0), K.Step("D", 1, 1, (), ((), (), (2,)), 0), K.Step("D", 2, 2, (), ((), (), ()), 0)))
    assert E.suggested_event(quiet) == 3
