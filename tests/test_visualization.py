"""Figuren: feste Achsen, Blockansicht (Farbe = wahre Reihenfolge, Zahl = Rang nach der Prognose), Kurven, Balken, Verteilung. Zahlen kommen aus künstlichen Eingaben, nicht aus dem Modell."""

import numpy as np
import pytest

import vwz_constants as C
import vwz_evaluation as E
import vwz_stk_core as K
import vwz_visualization as V

LOW, MED, Q70, HELL = C.RULE_LOWEST, C.RULE_MEDIAN, C.RULE_Q70, C.RULE_HELL


def axes_locked(fig):
    return bool(fig.layout.xaxis.fixedrange) and bool(fig.layout.yaxis.fixedrange)


def trace_named(fig, name):
    return next(t for t in fig.data if t.name == name)


# ---------------------------------------------------------------------------------------------------
# Blockansicht
# ---------------------------------------------------------------------------------------------------
def test_block_title_has_two_lines():
    assert V.block_title("Regel", 7) == "<b>Regel</b><br><sub>Umstapelungen bisher: 7</sub>"


def test_block_figure_colors_by_true_order_and_labels_by_estimated_rank():
    stacks = ((0, 1), (2,), ())
    t_dep = {0: 5.0, 1: 3.0, 2: 9.0}                       # wahre Reihenfolge: 1, 0, 2
    est = {0: 4.0, 1: 8.0, 2: 1.0}                         # geschätzte Reihenfolge: 2, 0, 1
    fig = V.block_figure(stacks, 3, t_dep, est, "T", moved=(1,), arrived=2, dwell={0: 5.0, 1: 3.0, 2: 9.0}, est_days={0: 4.0, 1: 8.0, 2: 1.0})
    text = fig.data[0]
    assert list(text.text) == ["2", "3", "1"]                                          # Zahl = Rang nach der Prognose, in Stapel-/Ebenenreihenfolge (0, 1, 2)
    assert len(text.hovertext) == 3 and "wahre Abfahrtsreihenfolge: 2 von 3" in text.hovertext[0] and "nach Prognose: 2 von 3" in text.hovertext[0]
    assert "gerade umgestapelt" in text.hovertext[1] and "gerade angekommen" in text.hovertext[2] and "wahr 5.0 d, prognostiziert 4.0 d" in text.hovertext[0]
    boxes = [s for s in fig.layout.shapes if s.fillcolor.startswith("rgb")]
    soon, late = C.BLOCK_COLOR_SOON, C.BLOCK_COLOR_LATE
    assert len(boxes) == 3
    assert boxes[1].fillcolor == f"rgb({soon[0]},{soon[1]},{soon[2]})" and boxes[2].fillcolor == f"rgb({late[0]},{late[1]},{late[2]})"                   # Container 1 fährt zuerst (dunkel), 2 zuletzt (hell)
    assert boxes[1].line.color == C.MOVED_COLOR and boxes[2].line.color == C.ARRIVED_COLOR and boxes[0].line.color is None
    assert len([s for s in fig.layout.shapes if s.fillcolor == C.BLOCK_STACK_BG]) == 3 and axes_locked(fig) and fig.layout.height == C.BLOCK_FIGURE_BASE_PX + 14 + 3 * C.BLOCK_FIGURE_TIER_PX
    assert list(text.textfont.color) == ["#1c2430", "white", "#1c2430"]               # dunkle Kästen (fahren bald ab): weiße Zahl


def test_block_figure_text_color_depends_on_the_shade():
    fig = V.block_figure(((0, 1, 2),), 3, {0: 1.0, 1: 2.0, 2: 3.0}, {0: 1.0, 1: 2.0, 2: 3.0}, "T")
    assert list(fig.data[0].textfont.color) == ["white", "#1c2430", "#1c2430"]        # dunkel (f unter 0,5): weiße Zahl; helle Kästen: dunkle Zahl (Mitte f = 0,5 zählt als hell)
    empty = V.block_figure(((), ()), 2, {}, {}, "leer")
    assert len(empty.data[0].x) == 0 and axes_locked(empty)


def test_block_figure_reads_a_real_block():
    p = E.Params(6, 5, 80, 100, 2, 3, "linear", 5)
    outs = E.run_rules(p, 3)
    blk = E.block_of(p, 3)
    step = E.suggested_event(outs[0].result)
    stacks, moved, arrived, _ = E.state_at(blk.inst, outs[0].result, step)
    fig = V.block_figure(stacks, 5, blk.t_dep, blk.t_arr + outs[2].est_days, "x", moved, arrived, blk.dwell, outs[2].est_days)
    assert len(fig.data[0].text) == sum(len(s) for s in stacks) and set(fig.data[0].text) <= {str(i) for i in range(1, 30)}


# ---------------------------------------------------------------------------------------------------
# Paarfehler-Kurve
# ---------------------------------------------------------------------------------------------------
def curve():
    return E.GaussCurve((0.5, 1.0, 2.0), (0.4, 0.8, 1.2), (0.4, 0.8, 1.6), (0.2, 0.3, 0.4), 0.9, 0.1, 10)


def test_pair_error_figure_traces_lines_and_marker():
    c = curve()
    tip = E.tipping_point(c)
    fig = V.pair_error_figure(c, tip, [(30, 0.35, 1.0), (1000, 0.21, 0.74)], (0.2, 0.75), "Gradient Boosting")
    gauss = trace_named(fig, "Gauß-Rauschen (Stapelplanung)")
    assert list(gauss.x) == pytest.approx([20, 30, 40]) and list(gauss.y) == [0.4, 0.8, 1.2] and [row[0] for row in gauss.customdata] == [0.5, 1.0, 2.0]
    assert list(trace_named(fig, "Hellseher").x) == [0] and list(trace_named(fig, "Hellseher").y) == [0.1]
    learned = trace_named(fig, "gelernt: Gradient Boosting, Quantil 0,7")
    assert list(learned.x) == pytest.approx([35, 21]) and list(learned.text) == ["30", "1.000"] and list(learned.y) == [1.0, 0.74]
    cur = trace_named(fig, "eingestellte Prognose")
    assert list(cur.x) == pytest.approx([20]) and list(cur.y) == [0.75] and cur.marker.line.color == C.MARKER_LINE_COLOR
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert sorted(s.y0 for s in lines if s.y0 == s.y1) == [0.9] and any(s.x0 == s.x1 and s.x0 == pytest.approx(tip.pair_err * 100) for s in lines)      # Ausgleich waagerecht, Kipppunkt senkrecht
    assert axes_locked(fig)


def test_pair_error_figure_without_learned_points_or_tipping():
    c = E.GaussCurve((0.5, 1.0, 2.0), (0.2, 0.3, 0.4), (0.4, 0.8, 1.6), (0.2, 0.3, 0.4), 0.9, 0.1, 10)              # Bestfit gewinnt überall: kein Kipppunkt
    fig = V.pair_error_figure(c, E.tipping_point(c))
    assert [t.name for t in fig.data] == ["Gauß-Rauschen (Stapelplanung)", "Hellseher"] and len([s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]) == 0
    assert axes_locked(fig)


# ---------------------------------------------------------------------------------------------------
# Lernkurve
# ---------------------------------------------------------------------------------------------------
def learning_curve():
    sizes = C.TRAIN_SIZES
    return E.LearningCurve(sizes, {"linear": (0.1, -0.05, -0.06, -0.07, -0.07, -0.08), "boosting": (0.12, 0.03, -0.07, -0.12, -0.13, -0.15), "gruppe": (0.24, 0.23, 0.21, 0.19, 0.2, 0.19)},
                           {"linear": ((0.0, 0.2),) + ((-0.05, -0.05),) * 5, "boosting": ((0.03, 0.2), (-0.05, 0.05), (-0.09, -0.03)) + ((-0.12, -0.12), (-0.13, -0.13), (-0.15, -0.15)),
                            "gruppe": ((0.2, 0.25),) + ((0.2, 0.2),) * 5}, (0.3, 0.28, 0.24, 0.21, 0.2, 0.19), (0.97, 0.9, 0.77, 0.74, 0.72, 0.7), 60, (5, 6, 7, 8))


def test_learning_curve_figure_lines_band_and_reference_lines():
    lc = learning_curve()
    fig = V.learning_curve_figure(lc, -0.6, "boosting", 1000)
    sel = trace_named(fig, "Gradient Boosting, Quantil 0,7 (eingestellt)")
    assert list(sel.x) == list(C.TRAIN_SIZES) and list(sel.y) == list(lc.series["boosting"]) and sel.line.width == 3
    other = trace_named(fig, "Lineare Regression, Quantil 0,7")
    assert other.line.width < 3 and trace_named(fig, "Gruppenmittel nach Art (Mittelwert je Art)").line.dash == "dot"
    band = next(t for t in fig.data if t.fill == "toself")
    assert list(band.x) == list(C.TRAIN_SIZES) + list(C.TRAIN_SIZES)[::-1] and max(band.y) == 0.2 and min(band.y) == -0.15                # Band des eingestellten Lerners
    cursor = trace_named(fig, "eingestellte Trainingsmenge")
    assert list(cursor.x) == [1000, 1000] and cursor.y[0] == pytest.approx(-0.6) and cursor.y[1] == pytest.approx(0.25)                    # von der tiefsten bis zur höchsten Größe (Hellseher, Gruppenmittel)
    hl = [s for s in fig.layout.shapes if s.type == "line"]
    assert sorted(s.y0 for s in hl) == [-0.6, 0.0] and fig.layout.xaxis.type == "log" and list(fig.layout.xaxis.tickvals) == list(C.TRAIN_SIZES) and fig.layout.xaxis.ticktext[-1] == "10.000"
    assert axes_locked(fig)
    lin = V.learning_curve_figure(lc, -0.6, "linear", 30)
    assert trace_named(lin, "Lineare Regression, Quantil 0,7 (eingestellt)").line.width == 3 and next(t for t in lin.data if t.fill == "toself").y[0] == 0.2


# ---------------------------------------------------------------------------------------------------
# Median oder Quantil, Verteilung
# ---------------------------------------------------------------------------------------------------
def test_forecast_kind_figure_labels_bars_with_the_mae_and_draws_the_reference():
    rows = [("Median", 0.78, 0.31, "#111111"), ("Mittelwert", 0.72, 0.32, "#222222"), ("Quantil 0,7", 0.74, 0.34, "#333333")]
    fig = V.forecast_kind_figure(rows, 0.86)
    bar = fig.data[0]
    assert list(bar.x) == ["Median<br>MAE 0.31", "Mittelwert<br>MAE 0.32", "Quantil 0,7<br>MAE 0.34"] and list(bar.y) == [0.78, 0.72, 0.74] and list(bar.text) == ["0.780", "0.720", "0.740"]
    assert list(bar.marker.color) == ["#111111", "#222222", "#333333"]
    assert [s.y0 for s in fig.layout.shapes if s.type == "line"] == [0.86] and tuple(fig.layout.yaxis.range) == (0, pytest.approx(0.86 * 1.25)) and axes_locked(fig)
    assert tuple(V.forecast_kind_figure([("A", 1.5, 0.1, "#111")], 0.5).layout.yaxis.range) == (0, pytest.approx(1.5 * 1.25))       # höchster Balken bestimmt die Achse


def test_distribution_figure_colors_bins_by_sign_and_marks_the_mean():
    diffs = [-0.30, -0.28, -0.12, -0.10, 0.02, 0.07, 0.08]
    fig = V.distribution_figure(diffs, -0.1, "gelernt")
    bar = fig.data[0]
    assert sum(bar.y) == pytest.approx(100.0) and len(bar.x) == len(bar.y) == len(bar.marker.color)
    for x, c in zip(bar.x, bar.marker.color):
        assert c == (C.RULE_COLORS[Q70] if x < 0 else C.MOVED_COLOR)
    assert bar.x[0] == pytest.approx(-0.275) and bar.x[-1] == pytest.approx(0.125) and all(abs((b - a) - 0.05) < 1e-9 for a, b in zip(bar.x, bar.x[1:]))      # Klassen der Breite 0,05 von -0,3 bis 0,15
    assert bar.y[0] == pytest.approx(200 / 7) and bar.y[1] == 0 and bar.y[3] == pytest.approx(100 / 7) and bar.y[4] == pytest.approx(100 / 7)                # -0,30 und -0,28 in der ersten Klasse, -0,12 und -0,10 in der vierten und fünften
    vlines = sorted(s.x0 for s in fig.layout.shapes if s.type == "line")
    assert vlines == [-0.1, 0.0] and "gelernt" in fig.layout.xaxis.title.text and axes_locked(fig)
    same = V.distribution_figure([-0.2, -0.2, -0.2], -0.2, "x")                        # alle gleich: ein Balken mit 100 %
    assert max(same.data[0].y) == pytest.approx(100.0) and axes_locked(same)


# ---------------------------------------------------------------------------------------------------
# Kumuliert, Streudiagramm, Regelbalken
# ---------------------------------------------------------------------------------------------------
def fake_outcomes():
    def res(cum):
        return K.Result(cum[-1], 3, 0, 0, tuple(cum), ())
    mk = lambda key, cum: E.Outcome(key, C.RULE_LABELS[key], res(cum), None, None)                    # noqa: E731
    return (mk(LOW, [0, 1, 1, 3]), mk(MED, [0, 0, 1, 2]), mk(Q70, [0, 0, 0, 1]))


def test_cumulative_figure_shows_reference_and_focus_with_a_cursor():
    outs = fake_outcomes()
    fig = V.cumulative_figure(outs, Q70, cursor=2)
    assert [t.name for t in fig.data] == ["Niedrigster Stapel", "Bestfit Quantil 0,7"] and list(fig.data[1].y) == [0, 0, 0, 0, 1] and list(fig.data[0].x) == [0, 1, 2, 3, 4]
    assert fig.data[0].line.color == C.RULE_COLORS[LOW] and fig.data[1].line.color == C.RULE_COLORS[Q70] and fig.data[0].line.shape == "hv"
    assert [s.x0 for s in fig.layout.shapes if s.type == "line"] == [2] and axes_locked(fig)
    only = V.cumulative_figure(outs, LOW)
    assert [t.name for t in only.data] == ["Niedrigster Stapel"] and not only.layout.shapes


def test_scatter_figure_has_a_diagonal_and_log_axes():
    fig = V.scatter_figure([1.0, 4.0, 9.0], [2.0, 4.0, 6.0], "Quantil 0,7")
    diag, pts = fig.data
    assert list(diag.x) == list(diag.y) and diag.x[0] < 1.0 and diag.x[1] > 9.0 and list(pts.x) == [2.0, 4.0, 6.0] and list(pts.y) == [1.0, 4.0, 9.0] and pts.name == "Quantil 0,7"
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log" and tuple(fig.layout.xaxis.range) == tuple(fig.layout.yaxis.range) and axes_locked(fig)
    lo, hi = fig.layout.xaxis.range
    assert lo == pytest.approx(np.log10(0.8 * 1.0)) and hi == pytest.approx(np.log10(9.0 * 1.25))


def test_rule_bar_figure_per_container_from_zero():
    outs = fake_outcomes()
    fig = V.rule_bar_figure(outs, 30)
    bar = fig.data[0]
    assert list(bar.x) == ["Niedrigster Stapel", "Bestfit Median", "Bestfit Quantil 0,7"] and list(bar.y) == pytest.approx([0.1, 2 / 30, 1 / 30]) and list(bar.text) == ["0.100", "0.067", "0.033"]
    assert tuple(fig.layout.yaxis.range)[0] == 0 and tuple(fig.layout.yaxis.range)[1] == pytest.approx(0.1 * 1.2 + 0.01) and axes_locked(fig)
    assert list(bar.marker.color) == [C.RULE_COLORS[LOW], C.RULE_COLORS[MED], C.RULE_COLORS[Q70]]
