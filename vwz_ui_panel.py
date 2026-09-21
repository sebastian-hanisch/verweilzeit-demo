"""Wiederverwendbare Bausteine zur Darstellung einer Regel: Kennzahlen im 2 x 2-Raster (Hauptansicht und Tabs) und das Panel je Regel im Regelvergleich."""

import streamlit as st

import vwz_constants as C
import vwz_evaluation as E
import vwz_visualization as V


def render_metrics(outcome, outcomes, n_containers):
    """Vier Kennzahlen im 2 x 2-Raster (vier Spalten schneiden die Namen bei 800 px ab). Nutzen (Umstapelungen) und Prognosegüte (MAE, Paarfehler) stehen getrennt, nie zu einer Zahl gemischt.
    Delta = diese Regel minus Bezug (Niedrigster Stapel ohne Prognose); weniger ist besser."""
    base = E.outcome_of(outcomes, C.BASELINE)
    is_base = outcome.key == base.key
    per, per_base = outcome.moves / n_containers, base.moves / n_containers
    rows = [st.columns(2), st.columns(2)]
    m = rows[0] + rows[1]
    m[0].metric("Umstapelungen je Container", f"{per:.3f}", delta=None if is_base else f"{per - per_base:+.3f}", delta_color="inverse",
                help=f"Am gezeigten Block: {outcome.moves} Umstapelungen bei {n_containers} Containern; Bezug (Niedrigster Stapel ohne Prognose): {base.moves}. Delta = Regel minus Bezug, weniger ist besser.")
    if is_base or not base.moves:
        m[1].metric("Ersparnis gegen Ausgleich", "–", help="Anteil der Umstapelungen, den die Regel gegenüber dem Ausgleich spart (negativ = mehr). Für den Bezug selbst leer.")
    else:
        m[1].metric("Ersparnis gegen Ausgleich", f"{(base.moves - outcome.moves) / base.moves * 100:+.0f} %",
                    help="Anteil der Umstapelungen, den die Regel am gezeigten Block gegenüber dem Ausgleich spart; negativ heißt: mehr Umstapelungen als der Ausgleich.")
    q = outcome.quality
    m[2].metric("Prognosefehler (MAE)", "–" if q is None else f"{q.mae:.2f} Standzeiten",
                help="Mittlerer Betragsfehler der prognostizierten Verweilzeit, in mittleren Standzeiten. Ohne Prognose leer. Ein kleiner MAE heißt nicht wenig Umstapeln (siehe Kernabschnitt).")
    m[3].metric("Falsch geordnete Paare", "–" if q is None else f"{q.pair_err * 100:.0f} %",
                help="Anteil der gleichzeitig anwesenden Containerpaare, deren Abfahrtsreihenfolge die Prognose falsch herum sieht. Das ist das Maß, an dem der Nutzen hängt.")


def render_rule_panel(prefix, outcome, outcomes, n_containers, dwell=None):
    """Beschreibung, Kennzahlen (2 x 2) und kumulierte Umstapelkurve einer Regel; bei der Quantil-Regel dazu das Streudiagramm Prognose gegen wahre Verweilzeit. `outcomes` müssen mit record=True
    gerechnet sein."""
    st.markdown(C.RULE_DESCRIPTIONS[outcome.key])
    render_metrics(outcome, outcomes, n_containers)
    st.plotly_chart(V.cumulative_figure(outcomes, outcome.key), width="stretch", key=f"{prefix}_cumulative_chart")
    if outcome.key == C.RULE_Q70 and outcome.est_days is not None and dwell is not None:
        st.plotly_chart(V.scatter_figure(outcome.est_days, dwell, "Quantil 0,7"), width="stretch", key=f"{prefix}_scatter_chart")
        st.caption("Jeder Punkt ist ein Container des gezeigten Blocks: Prognose (Quantil 0,7) gegen wahre Verweilzeit, beide logarithmisch. Auf der Diagonale liegt eine richtige Prognose; "
                   "das vorsichtige Quantil liegt im Mittel darüber, und die Streuung um die Diagonale ist das, was die Regel falsch ordnet.")


def event_control(n_events):
    """Schrittregler über die Ereignisse des Blocks (0 = leerer Block). Die Grenze ist berechnet: hat der Block keine Ereignisse, wäre min == max, und Streamlit lehnt den Regler ab; dann ein Hinweis."""
    if n_events < 1:
        st.caption("Der Block hat keine Ereignisse, kein Regler nötig.")
        return 0
    return st.slider("Ereignis", 0, n_events, key="event_slider",
                     help="Ein Ereignis ist eine Ankunft oder eine Abholung. Startpunkt: eine Abholung mit Umstapelung bei hoher Belegung (Bezug).")
