"""Bausteine der Oberfläche: Kennzahlen im 2 x 2-Raster (mit Deltas gegen den Bezug), Panel je Regel, Schrittregler mit berechneter Grenze (auch der Randfall ohne Ereignisse)."""

from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest


def metrics_app():
    import streamlit as st

    import vwz_constants as C
    import vwz_evaluation as E
    import vwz_stk_core as K
    from vwz_ui_panel import render_metrics

    def outcome(key, moves, quality=None):
        return E.Outcome(key, C.RULE_LABELS[key], K.Result(moves, 300, 0, 0, (), ()), quality, None)

    base = outcome(C.RULE_LOWEST, 240)
    q70 = outcome(C.RULE_Q70, 216, E.Quality(0.34, 0.214))
    worse = outcome(C.RULE_MEDIAN, 264, E.Quality(0.31, 0.228))
    zero = outcome(C.RULE_LOWEST, 0)
    outs = (base, q70, worse)
    st.markdown("BASE")
    render_metrics(base, outs, 300)
    st.markdown("Q70")
    render_metrics(q70, outs, 300)
    st.markdown("WORSE")
    render_metrics(worse, outs, 300)
    st.markdown("ZERO")
    render_metrics(outcome(C.RULE_Q70, 3, E.Quality(0.1, 0.0)), (zero, outcome(C.RULE_Q70, 3, E.Quality(0.1, 0.0))), 300)


def test_metrics_grid_values_deltas_and_colors():
    at = AppTest.from_function(metrics_app, default_timeout=60).run()
    assert not at.exception, at.exception
    m = [(x.label, x.value, x.delta) for x in at.metric]
    assert len(m) == 16 and [x[0] for x in m[:4]] == ["Umstapelungen je Container", "Ersparnis gegen Ausgleich", "Prognosefehler (MAE)", "Falsch geordnete Paare"]
    assert m[:4] == [("Umstapelungen je Container", "0.800", ""), ("Ersparnis gegen Ausgleich", "–", ""), ("Prognosefehler (MAE)", "–", ""), ("Falsch geordnete Paare", "–", "")]           # Bezug: kein Delta
    assert m[4:8] == [("Umstapelungen je Container", "0.720", "-0.080"), ("Ersparnis gegen Ausgleich", "+10 %", ""), ("Prognosefehler (MAE)", "0.34 Standzeiten", ""), ("Falsch geordnete Paare", "21 %", "")]
    assert m[8:12] == [("Umstapelungen je Container", "0.880", "+0.080"), ("Ersparnis gegen Ausgleich", "-10 %", ""), ("Prognosefehler (MAE)", "0.31 Standzeiten", ""), ("Falsch geordnete Paare", "23 %", "")]
    assert at.metric[4].proto.color == MetricProto.GREEN and at.metric[8].proto.color == MetricProto.RED                          # weniger Umstapelungen = grün (invers)
    assert m[13][1] == "–" and m[12][2] == "+0.010"                                                                                 # Bezug ohne Umstapelungen: keine Ersparnis in Prozent


def event_app():
    import streamlit as st

    from vwz_ui_panel import event_control
    st.session_state.setdefault("event_slider", 3)
    st.write(f"EVENT={event_control(st.session_state['n'])}")


def test_event_control_slider_and_the_degenerate_case_without_events():
    at = AppTest.from_function(event_app, default_timeout=60)
    at.session_state["n"] = 5
    at.run()
    assert not at.exception and at.slider(key="event_slider").value == 3 and (at.slider(key="event_slider").min, at.slider(key="event_slider").max) == (0, 5) and "EVENT=3" in at.markdown[0].value
    at = AppTest.from_function(event_app, default_timeout=60)
    at.session_state["n"] = 0
    at.run()
    assert not at.exception and len(at.slider) == 0 and "EVENT=0" in at.markdown[0].value                                       # min == max lehnt Streamlit ab: stattdessen ein Hinweis
    assert any("kein Regler nötig" in c.value for c in at.caption)


def panel_app():
    import streamlit as st

    import vwz_constants as C
    import vwz_evaluation as E
    from vwz_ui_panel import render_rule_panel

    p = E.Params(6, 5, 80, 100, 2, 3, "linear", 5)
    outs = E.run_rules(p, 3)
    blk = E.block_of(p, 3)
    for key in C.TAB_RULES:
        render_rule_panel(f"rule_{key}", next(o for o in outs if o.key == key), outs, 100, dwell=blk.dwell)
    render_rule_panel("nodwell", next(o for o in outs if o.key == C.RULE_Q70), outs, 100)
    st.write("done")


def test_rule_panel_charts_and_the_scatter_only_for_the_quantile_rule():
    at = AppTest.from_function(panel_app, default_timeout=120).run()
    assert not at.exception, at.exception
    keys = [c.key for c in at.get("plotly_chart")]
    assert keys == ["rule_niedrigster_stapel_cumulative_chart", "rule_bestfit_median_cumulative_chart", "rule_bestfit_q70_cumulative_chart", "rule_bestfit_q70_scatter_chart",
                    "rule_unsicherheitsbewusst_cumulative_chart", "nodwell_cumulative_chart"]                                     # Streudiagramm nur beim Quantil und nur mit Verweilzeiten
    assert len(at.metric) == 5 * 4 and any("Jeder Punkt ist ein Container" in c.value for c in at.caption)
    assert sum("Bestfit auf Ankunft plus **Quantil 0,7**" in m.value for m in at.markdown) == 2
