"""AppTest: Skelett und Footer, jedes Preset, Permalink, alle Regler an Min und Max (jeder Lerner mit und ohne Ankündigung), Kennzahlen im 2 x 2-Raster, die bedingte Meldung in allen Zuständen,
Kernabschnitt (Kipppunkt, Urteil, Lernkurve), Regelvergleich, PDF, Texte."""

import pathlib

import pytest
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest

import vwz_constants as C
import vwz_evaluation as E
from vwz_presets import SETTING_SPECS

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
FOOTER = ("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
          "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
          "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)")

PINS = {"Nur Standard": ("0.840", "+0.047", "-6 %"), "Mit Ankündigung": ("0.740", "-0.053", "+7 %"), "Zu wenig Daten": ("0.937", "+0.143", "-18 %"), "Sehr verlässlich": ("0.533", "-0.260", "+33 %"),
        "Voller Block": ("1.033", "-0.150", "+13 %")}


@pytest.fixture(autouse=True)
def clean_cache():
    """st.cache_data ist prozessweit: Tests, die Funktionen ersetzen, dürfen keine zwischengespeicherten Ergebnisse anderer Tests sehen."""
    import streamlit as st
    st.cache_data.clear()
    yield


def fresh(**query):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def set_and_run(at, **values):
    for key, value in values.items():
        if key.endswith("_input"):
            at.number_input(key=key).set_value(value)
        elif key in ("train_slider", "announce_slider"):
            at.select_slider(key=key).set_value(value)
        elif key == "learner_select":
            at.selectbox(key=key).set_value(value)
        elif key == "view_radio":
            at.radio(key=key).set_value(value)
        else:
            at.slider(key=key).set_value(value)
    at.run()
    assert not at.exception, at.exception
    return at


def main_metrics(at):
    return [(m.label, m.value, m.delta) for m in at.metric[:4]]


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def message(at, needle):
    for group in (at.success, at.warning, at.info):
        for x in group:
            if needle in x.value:
                return x
    return None


# ---------------------------------------------------------------------------------------------------
# Skelett
# ---------------------------------------------------------------------------------------------------
def test_skeleton_and_footer():
    at = fresh()
    assert [h.value for h in at.sidebar.header] == ["⚙️ Einstellungen"]                # genau EIN Header
    assert len(at.title) == 1 and "Verweilzeit" in at.title[0].value
    assert any(v.value.startswith("## 🎯") for v in at.markdown)
    assert [s.value for s in at.subheader] == ["📐 Wie gut muss die Prognose sein, und wie viele Daten braucht sie?"]
    assert [e.label for e in at.expander] == ["🔧 Wie wir das erreichen – Regeln im Vergleich", "Wie funktioniert diese Demo?", "📐 Mathematische Formulierung"]
    assert any(c.value == FOOTER for c in at.caption)
    presets = [b.label for b in at.button if b.label in C.PRESETS]
    assert presets == list(C.PRESETS) and len(presets) == 5 and all(len(n) <= 16 for n in presets)
    assert [s.label for s in at.sidebar.slider] == ["Anzahl Stapel", "Maximale Stapelhöhe", "Füllgrad des Blocks (%)", "Anzahl Container (Durchsatz)"]
    assert [s.label for s in at.sidebar.select_slider] == ["Trainingscontainer", "Verlässlichkeit der Ankündigung"] and [s.label for s in at.sidebar.selectbox] == ["Lerner"]
    assert [n.label for n in at.sidebar.number_input] == ["Zufalls-Seed (Block)", "Trainings-Seed"] and any(b.label == "🎲 Neuer Block" for b in at.sidebar.button)
    assert list(at.selectbox(key="learner_select").options) == ["Lineare Regression", "Gradient Boosting"]                     # das Gruppenmittel ist keine Wahl (toter Regler)
    assert list(at.select_slider(key="announce_slider").options) == list(C.NU_LABELS) and list(at.select_slider(key="train_slider").options) == ["30", "100", "300", "1.000", "3.000", "10.000"]
    assert [m.label for m in at.radio] == ["Rechts vergleichen mit"] and list(at.radio(key="view_radio").options) == [C.RULE_LABELS[k] for k in C.VIEW_KEYS]


def test_main_metrics_are_2x2_with_signed_deltas_against_the_reference():
    at = fresh()
    assert [m[0] for m in main_metrics(at)] == ["Umstapelungen je Container", "Ersparnis gegen Ausgleich", "Prognosefehler (MAE)", "Falsch geordnete Paare"]
    assert [m[1] for m in main_metrics(at)] == ["0.740", "+7 %", "0.31 Standzeiten", "20 %"] and [m[2] for m in main_metrics(at)] == ["-0.053", "", "", ""]
    assert at.metric[0].proto.color == MetricProto.GREEN and all(len(m[0]) <= 28 for m in main_metrics(at))                    # weniger Umstapelungen = besser = grün ("inverse")


def test_caption_states_block_capacity_and_the_learner():
    at = fresh()
    cap = [c.value for c in at.caption if c.value.startswith("Block 6 × 5")][0]
    assert cap.startswith("Block 6 × 5 mit höchstens 20 Containern gleichzeitig, 300 Container, Ankündigung: mittel (ν 0,6), Gradient Boosting mit 1.000 Trainingscontainern.")
    assert "Delta = Regel minus Niedrigster Stapel ohne Prognose" in cap
    at = fresh(fp="100", lr="linear", tn="5", an="0")
    cap = [c.value for c in at.caption if c.value.startswith("Block 6 × 5")][0]
    assert "höchstens 25 Containern" in cap and "Ankündigung: keine, Lineare Regression mit 10.000 Trainingscontainern" in cap


def test_core_section_metrics_and_bases():
    at = fresh()
    assert [(m.label, m.value) for m in at.metric[4:7]] == [("Kipppunkt (Paare)", "28 %"), ("Ihr Paarfehler", "21 %"), ("Wirkt wie σ", "0.48")]
    caps = [c.value for c in at.caption]
    assert any("Basis: 60 Blöcke (Seeds 0-59, nicht Ihr Seed) mit Ihren Einstellungen. Grau das Gauß-Rauschen der Stapelplanung" in c and "der Hellseher (0.250)" in c for c in caps)
    assert any("Basis: 60 Blöcke (Seeds 0-59, nicht Ihr Seed). Besser an 88 %" in c for c in caps)
    assert any("Basis: 60 Blöcke (Seeds 0-59); Differenz zum Ausgleich je Container" in c and "über 4 Trainings-Seeds (ab Ihrem Seed 5)" in c for c in caps)
    assert any("Basis: 60 Blöcke, derselbe Lerner (Gradient Boosting)" in c for c in caps)


def test_charts_are_present_with_unique_keys():
    at = fresh()
    keys = [c.key for c in at.get("plotly_chart")]
    assert len(keys) == 12 and len(set(keys)) == 12 and all(keys)                       # 2 Blöcke, Paarfehler, Verteilung, Lernkurve, Median/Quantil, 4 Regel-Tabs + Streudiagramm, Vergleich
    assert keys[:6] == ["block_chart_left", "block_chart_right", "pair_error_chart", "distribution_chart", "learning_curve_chart", "forecast_kind_chart"]


# ---------------------------------------------------------------------------------------------------
# Presets, Permalink
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_within_widget_bounds_and_shows_its_story(name):
    at = fresh()
    click(at, name)
    p = C.PRESETS[name]
    assert at.slider(key="fill_slider").value == p["fill_pct"] and at.slider(key="n_containers_slider").value == p["n_containers"]
    assert at.select_slider(key="announce_slider").value == p["announce"] and at.select_slider(key="train_slider").value == p["train_idx"]
    assert at.selectbox(key="learner_select").value == p["learner"] and at.number_input(key="seed_input").value == p["seed"] and at.number_input(key="train_seed_input").value == p["train_seed"]
    for state_key, spec in SETTING_SPECS.items():
        if spec.lo is not None:
            value = at.session_state[state_key]
            assert spec.lo <= value <= spec.hi and (spec.step in (None, 1) or (value - spec.lo) % spec.step == 0)
    per, delta, saving = PINS[name]                                                                                            # Umstapelungen am gezeigten Block (Messreihe: 252 / 222 / 281 / 160 / 310 von 300 Containern)
    assert (main_metrics(at)[0][1], main_metrics(at)[0][2], main_metrics(at)[1][1]) == (per, delta, saving)
    story = "Die Prognose kippt" if name in ("Nur Standard", "Zu wenig Daten") else "Die Prognose lohnt sich"               # die Geschichte des Presets (Urteil an der Stichprobe)
    assert message(at, story) is not None


def test_permalink_is_clamped_snapped_and_ignores_garbage():
    at = fresh(fp="83", nc="320", an="9", tn="-2", lr="junk", vw="niedrigster_stapel", ns="abc", ts="500")
    assert at.slider(key="fill_slider").value == 85 and at.slider(key="n_containers_slider").value == 300 and at.select_slider(key="announce_slider").value == 4
    assert at.select_slider(key="train_slider").value == 0 and at.selectbox(key="learner_select").value == "boosting" and at.radio(key="view_radio").value == C.VIEW_DEFAULT
    assert at.slider(key="n_stacks_slider").value == C.N_STACKS_DEFAULT and at.number_input(key="train_seed_input").value == 99


def test_permalink_roundtrip_reflects_settings():
    at = fresh(ns="5", mh="4", fp="70", nc="200", an="3", tn="2", lr="linear", seed="11", ts="7", vw="unsicherheitsbewusst")
    values = {k: at.session_state[k] for k in SETTING_SPECS}
    assert values == {"n_stacks_slider": 5, "max_height_slider": 4, "fill_slider": 70, "n_containers_slider": 200, "announce_slider": 3, "train_slider": 2, "learner_select": "linear", "seed_input": 11,
                      "train_seed_input": 7, "view_radio": "unsicherheitsbewusst"}
    for key, spec in SETTING_SPECS.items():
        got = at.query_params[spec.url_param]
        got = got[0] if isinstance(got, list) else got
        assert got == spec.encoder(values[key]), key


def test_new_block_button_changes_only_the_seed():
    at = fresh()
    before = {k: at.session_state[k] for k in SETTING_SPECS if k != "seed_input"}
    click(at, "🎲 Neuer Block")
    assert {k: at.session_state[k] for k in SETTING_SPECS if k != "seed_input"} == before and 0 <= at.session_state["seed_input"] <= 9999


# ---------------------------------------------------------------------------------------------------
# Bedingte Meldung
# ---------------------------------------------------------------------------------------------------
def test_message_helps_is_a_success_with_the_block_numbers():
    at = fresh()
    msg = message(at, "Die Prognose lohnt sich")
    assert msg in list(at.success) and "Bestfit Quantil 0,7 braucht im Mittel über 60 Blöcke 14 % weniger Umstapelungen als der Ausgleich" in msg.value
    assert "Am gezeigten Block (Seed 268): 222 gegen 238 Umstapelungen" in msg.value and "Nur Standardmerkmale" not in msg.value and "Weniger als 300 Trainingscontainer" not in msg.value


def test_message_tipped_with_the_standard_features_addition():
    at = click(fresh(), "Nur Standard")
    msg = message(at, "Die Prognose kippt")
    assert msg in list(at.warning) and "Nur Standardmerkmale (Art, Kunde, Zoll, Wochenende, Vorlauf)" in msg.value and "Erst eine verlässliche Ankündigung trägt." in msg.value
    assert "Am gezeigten Block (Seed 268): 252 gegen 238 Umstapelungen" in msg.value and "Weniger als 300 Trainingscontainer" not in msg.value


def test_message_tipped_with_the_few_data_addition():
    at = click(fresh(), "Zu wenig Daten")
    msg = message(at, "Die Prognose kippt")
    assert msg in list(at.warning) and "Weniger als 300 Trainingscontainer" in msg.value and "Nur Standardmerkmale" not in msg.value and "281 gegen 238" in msg.value
    both = set_and_run(fresh(), announce_slider=0, train_slider=0)
    assert "Nur Standardmerkmale" in message(both, "Die Prognose").value and "Weniger als 300 Trainingscontainer" in message(both, "Die Prognose").value


def _fake_verdict(monkeypatch, kind, pct):
    monkeypatch.setattr(E, "verdict", lambda rows, key, ref=C.BASELINE: E.Verdict(kind, -0.2 if kind == "better" else 0.2 if kind == "worse" else 0.01, 0.05, pct, 60))


@pytest.mark.parametrize("kind,pct,fragment,group", [("better", -40.0, "40 % weniger Umstapelungen", "success"), ("better", None, "0.200 weniger Umstapelungen", "success"),
                                                     ("worse", 25.0, "25 % mehr Umstapelungen", "warning"), ("worse", None, "0.200 mehr Umstapelungen", "warning"),
                                                     ("unclear", 1.0, "Kein klarer Unterschied", "info")])
def test_message_and_verdict_sentences_in_all_variants(monkeypatch, kind, pct, fragment, group):
    _fake_verdict(monkeypatch, kind, pct)
    at = fresh()
    top = [x.value.replace("**", "") for x in getattr(at, group) if fragment in x.value.replace("**", "")]
    assert len(top) == 2                                                              # die Meldung der Hauptansicht und das Urteil des Kernabschnitts (dort fett gesetzt)
    assert all(t.count("(") == t.count(")") for t in top)


def test_the_message_follows_the_chosen_rule():
    at = set_and_run(fresh(an="0"), view_radio=C.RULE_AWARE)
    msg = message(at, "Die Prognose kippt: Unsicherheitsbewusst")
    assert msg is not None and main_metrics(at)[0][1:] == ("0.983", "+0.190") and main_metrics(at)[1][1] == "-24 %"
    assert at.query_params["vw"] in ("unsicherheitsbewusst", ["unsicherheitsbewusst"])
    at = set_and_run(fresh(), view_radio=C.RULE_MEDIAN)
    assert message(at, "Die Prognose lohnt sich: Bestfit Median") is not None and main_metrics(at)[0][1:] == ("0.767", "-0.027")


# ---------------------------------------------------------------------------------------------------
# Regler an den Grenzen
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key,value", [("n_stacks_slider", 3), ("n_stacks_slider", 8), ("max_height_slider", 3), ("max_height_slider", 6), ("fill_slider", 40), ("fill_slider", 100),
                                       ("n_containers_slider", 100), ("n_containers_slider", 500), ("train_slider", 0), ("train_slider", 5), ("announce_slider", 0), ("announce_slider", 4),
                                       ("learner_select", "linear"), ("seed_input", 0), ("seed_input", 9999), ("train_seed_input", 0), ("train_seed_input", 99)])
def test_every_control_works_at_its_minimum_and_maximum(key, value):
    at = set_and_run(fresh() if key == "n_containers_slider" else fresh(nc="100"), **{key: value})               # kleiner Block als Ausgangspunkt: schneller, die Randwerte bleiben dieselben
    assert at.session_state[key] == value and len(at.metric) >= 7 and len(at.get("plotly_chart")) == 12


@pytest.mark.parametrize("learner", list(C.LEARNERS))
@pytest.mark.parametrize("announce,train", [(0, 0), (0, 5), (4, 0), (4, 5)])
def test_every_learner_with_and_without_announcement_at_the_training_extremes(learner, announce, train):
    at = fresh(lr=learner, an=str(announce), tn=str(train), nc="100")
    assert not at.exception and (message(at, "Die Prognose") is not None or message(at, "Kein klarer Unterschied") is not None)
    assert len(at.metric) >= 7 and at.selectbox(key="learner_select").value == learner and len(at.get("plotly_chart")) == 12


def test_extreme_combinations_run_without_exception():
    at = fresh(ns="3", mh="3", fp="40", nc="100", an="0", tn="0", lr="linear")
    assert not at.exception and len(at.get("plotly_chart")) == 12
    at = fresh(ns="8", mh="6", fp="100", nc="500", an="4", tn="5", lr="boosting")
    assert not at.exception and len(at.get("plotly_chart")) == 12


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt
# ---------------------------------------------------------------------------------------------------
def test_the_sample_does_not_depend_on_the_block_seed():
    at = fresh()
    before = [c.value for c in at.caption if c.value.startswith("Basis: 60 Blöcke (Seeds 0-59, nicht Ihr Seed). Besser")]
    at = set_and_run(at, seed_input=5)
    assert [c.value for c in at.caption if c.value.startswith("Basis: 60 Blöcke (Seeds 0-59, nicht Ihr Seed). Besser")] == before and len(before) == 1


def test_the_learner_changes_the_sample_and_the_learning_curve_highlights_it():
    b = fresh()
    a = set_and_run(fresh(), learner_select="linear")
    assert main_metrics(a) != main_metrics(b) and [m.value for m in a.metric[4:7]] != [m.value for m in b.metric[4:7]]


def test_tipping_metric_without_a_crossing_shows_a_dash():
    at = fresh(ns="3", mh="3", fp="40", nc="100")                                         # kleiner, leerer Block: Bestfit gewinnt womöglich überall
    assert not at.exception and at.metric[4].label == "Kipppunkt (Paare)"
    assert at.metric[4].value == "–" or at.metric[4].value.endswith(" %")                 # ohne Kipppunkt ein Strich statt einer erfundenen Zahl


# ---------------------------------------------------------------------------------------------------
# Blick in den Block
# ---------------------------------------------------------------------------------------------------
def test_view_radio_switches_the_rule_shown_in_the_metrics_and_the_block():
    at = fresh()
    caps = [c.value for c in at.caption]
    assert any(c.startswith("Links der Bezug, rechts 🛡️ Bestfit vorsichtig (Quantil 0,7). Farbe = wahre Abfahrtsreihenfolge") for c in caps)
    at = at.radio(key="view_radio").set_value(C.RULE_MEDIAN).run()
    assert not at.exception and any(c.value.startswith("Links der Bezug, rechts 🎯 Bestfit mit Median.") for c in at.caption)


def test_event_slider_starts_at_a_relocation_and_follows_the_scenario():
    at = fresh()
    outs = E.run_rules(E.params_from_preset(C.PRESETS["Mit Ankündigung"]), 268)
    start = E.suggested_event(E.outcome_of(outs, C.BASELINE).result)
    assert at.slider(key="event_slider").value == start and (at.slider(key="event_slider").min, at.slider(key="event_slider").max) == (0, 600)
    at = at.slider(key="event_slider").set_value(0).run()
    assert not at.exception and sum(c.value == "Ereignis 0 von 600: Der Block ist noch leer." for c in at.caption) == 2
    at = set_and_run(at, seed_input=5)
    assert at.slider(key="event_slider").value != 0 and at.slider(key="event_slider").max == 600                       # neues Szenario: der Regler springt zum Startpunkt


# ---------------------------------------------------------------------------------------------------
# Regelvergleich, PDF, Texte
# ---------------------------------------------------------------------------------------------------
def test_comparison_table_has_a_row_per_rule_with_the_right_cells():
    at = fresh()
    df = at.dataframe[0].value
    assert list(df["Regel"]) == [C.RULE_SHORT[k] for k in C.RULE_KEYS]
    assert list(df["Umstapelungen je Container (Block)"]) == [0.793, 0.767, 0.74, 0.807, 0.717, 0.877, 0.263] and df["Delta gegen Ausgleich (Block)"].isna().tolist() == [True] + [False] * 6
    assert list(df["Delta gegen Ausgleich (Block)"])[1:] == [-0.027, -0.053, 0.013, -0.077, 0.083, -0.53]
    assert list(df["Delta gegen Ausgleich (Stichprobe ± Standardfehler)"])[:3] == ["–", "-0.075 ± 0.011", "-0.121 ± 0.011"] and list(df["MAE (Stichprobe)"])[6] == 0.0
    assert df["MAE (Stichprobe)"].isna().tolist() == [True] + [False] * 6 and list(df["Falsch geordnete Paare % (Stichprobe)"])[2] == 21.0


def test_each_rule_tab_shows_its_metrics_with_deltas_against_the_reference():
    at = fresh()
    tab = at.metric[7:]
    assert [m.label for m in tab] == ["Umstapelungen je Container", "Ersparnis gegen Ausgleich", "Prognosefehler (MAE)", "Falsch geordnete Paare"] * 4
    assert [m.value for m in tab[:4]] == ["0.793", "–", "–", "–"] and [m.delta for m in tab[:4]] == ["", "", "", ""]                # Bezug ohne Delta
    assert [m.value for m in tab[8:12]] == ["0.740", "+7 %", "0.31 Standzeiten", "20 %"] and tab[8].delta == "-0.053"     # MAE und Paarfehler am gezeigten Block (die Stichprobe hat 0.34 und 21 %)


def test_pdf_download_button_is_offered():
    at = fresh()
    buttons = at.get("download_button")
    assert len(buttons) == 1 and buttons[0].proto.label == "📄 Ergebnis als PDF herunterladen"


def test_texts_state_the_model_the_rules_the_assumptions_and_the_limits():
    at = fresh()
    text = "\n".join(m.value for m in at.expander[1].markdown)
    for needle in ("synthetische Strom", "erfunden", "Ankündigung", "Lineare Regression", "Gradient Boosting", "Gruppenmittel nach Art", "Median", "Quantil 0,7", "Niedrigster Stapel", "Bezug", "Paarfehler",
                   "Glücksspiel", "unerklärte Streuung", "keine Bündelung nach Schiff", "kein Geldbetrag", "Größenordnungen aus einer Simulation"):
        assert needle in text, needle
    math = "\n".join(m.value for m in at.expander[2].markdown)
    for needle in ("LogNormal", "0{,}524", "Pinball-Verlust", "Paarfehler", "Gauß-Hermite", "2\\,\\mathrm{SE}", "Bayes-Orakel"):
        assert needle in math, needle
    assert "](" not in text and "](" not in math                                          # keine toten Datei-Links in der App
    intro = at.markdown[0].value
    assert "Stapelplanung-Demo" in intro and 'Expander "Wie funktioniert diese Demo?"' in intro and '"📐 Mathematische Formulierung"' in intro


def test_every_preset_help_and_the_intro_are_present():
    at = fresh()
    helps = {b.label: b.help for b in at.button if b.label in C.PRESETS}
    assert len(helps) == 5 and all(h for h in helps.values()) and "30 Trainingscontainer" in helps["Zu wenig Daten"] and "ν 0,3" in helps["Sehr verlässlich"]
