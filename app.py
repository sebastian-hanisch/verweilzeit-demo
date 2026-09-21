"""
Verweilzeit lernen statt annehmen – interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zusatz zur Hafen-Linie (Stapelplanung, Predict-then-Optimize): Die geschätzte Abfahrt eines Containers wird aus Merkmalen gelernt (synthetische Daten, lineare Regression, Gradient Boosting) und in die
Einlagerungsregel gesteckt. Gezeigt wird, wie gut die Prognose sein muss, wie viele Trainingsdaten sie braucht und was sie in Umstapelungen wert ist, gegen den einfachen Ausgleich ohne Prognose.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import vwz_constants as C
import vwz_evaluation as E
import vwz_stk_core as K
import vwz_visualization as V
from vwz_pdf_export import generate_vwz_pdf
from vwz_presets import SETTING_SPECS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from vwz_ui_panel import event_control, render_metrics, render_rule_panel

st.set_page_config(page_title="Verweilzeit lernen – Sebastian Hanisch", layout="wide")

SCENARIO_KEYS = list(SETTING_SPECS)
LABEL = C.RULE_LABELS
BASE, MED, Q70, AWARE, MEAN, HELL = C.BASELINE, C.RULE_MEDIAN, C.RULE_Q70, C.RULE_AWARE, C.RULE_MEAN, C.RULE_HELL


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_block(key):
    """Ein Block (Seed) mit allen Regeln, mit Verlauf für die Darstellung."""
    return E.run_rules(E.Params(*key[:-1]), key[-1], record=True)


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_sample(key):
    """Stichprobe (Seeds 0-59), unabhängig vom eingestellten Seed."""
    return E.sample(E.Params(*key))


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_gauss(block_key):
    """Gauß-Kurve der Stapelplanung: hängt nur vom Block ab, nicht von Ankündigung, Lerner und Training."""
    return E.gauss_curve(block_key)


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_learning_curve(key):
    """Lernkurve: läuft über die Trainingsmenge, hängt also nicht von ihr ab."""
    return E.learning_curve(E.Params(*key))


st.title("🔮 Verweilzeit lernen statt annehmen")
st.markdown(
    """
Ein Container kommt an, und man weiß nur ungefähr, wann er abgeholt wird. In der Stapelplanung-Demo war diese Schätzung ein Regler; hier wird die **Verweilzeit** aus Merkmalen **gelernt**
(synthetische Daten, lineare Regression und Boosting) und als **Prognose** in die Einlagerungsregel gesteckt. Die Demo zeigt, wie gut die Prognose sein muss, wie viele Trainingsdaten sie braucht und ob sie
die **Umstapelungen** gegenüber dem einfachen Ausgleich senkt. Die Antwort ist nicht immer ja. Wie das Modell funktioniert, steht im Expander "Wie funktioniert diese Demo?" weiter unten, die formale
Beschreibung im Expander "📐 Mathematische Formulierung".
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Nur Standard": "Keine Ankündigung: aus Art, Kunde, Zoll, Wochenende und Vorlauf lässt sich die Verweilzeit nicht gut genug lernen, der einfache Ausgleich ist besser.",
    "Mit Ankündigung": "Eine mittel verlässliche Ankündigung (Abholtermin) trägt die Prognose: weniger Umstapelungen als der Ausgleich.",
    "Zu wenig Daten": "Dieselbe Ankündigung, aber nur 30 Trainingscontainer: das Modell lernt zu wenig und ist schlechter als der Ausgleich.",
    "Sehr verlässlich": "Eine gute Ankündigung (ν 0,3): fast alles Wesentliche steckt in ihr, knapp ein Drittel weniger Umstapelungen.",
    "Voller Block": "Der Block ist bis an die Grenze gefüllt: alles ist teurer, und die Prognose spart auch hier deutlich.",
}
# Je Zeile drei Schaltflächen: bei fünf in einer Zeile werden die Namen in schmalen Fenstern abgeschnitten.
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(3)
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
cap_now = K.capacity_for(ss["n_stacks_slider"], ss["max_height_slider"], ss["fill_slider"] / 100)

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stacks = st.slider("Anzahl Stapel", *bounds("n_stacks_slider"), key="n_stacks_slider")
    max_height = st.slider("Maximale Stapelhöhe", *bounds("max_height_slider"), key="max_height_slider")
    fill_pct = st.slider("Füllgrad des Blocks (%)", *bounds("fill_slider"), step=C.FILL_PCT_STEP, format="%d%%", key="fill_slider",
                         help=f"Belegungsgrenze in Prozent von (Stapel - 1) mal Höhe; jetzt {cap_now} Container gleichzeitig. Ein Stapel bleibt als Platz zum Umstapeln frei. Je voller, desto teurer ist alles.")
    n_containers = st.slider("Anzahl Container (Durchsatz)", *bounds("n_containers_slider"), step=C.N_CONTAINERS_STEP, key="n_containers_slider",
                             help="Wie viele Container über den ganzen Ablauf ankommen und abgeholt werden. Standard 300 statt 120 der Stapelplanung: bei 120 sind Anlauf und Auslauf ein großer Teil "
                                  "des Laufs, das Urteil am einzelnen Block schwankt stärker (Messreihe).")
    st.markdown("**Prognose (Lernen)**")
    train_idx = st.select_slider("Trainingscontainer", options=list(range(C.TRAIN_RANGE[1] + 1)), format_func=lambda i: f"{C.TRAIN_SIZES[i]:,}".replace(",", "."), key="train_slider",
                                 help="Größe der Historie, aus der der Lerner lernt (abgeschlossene Container). Das Training läuft einmal je Einstellung und wird gemerkt; das Boosting trainiert fünf Modelle, bei 10.000 Containern gemessen rund 2 Sekunden.")
    announce = st.select_slider("Verlässlichkeit der Ankündigung", options=list(range(C.ANNOUNCE_RANGE[1] + 1)), format_func=lambda i: C.NU_LABELS[i], key="announce_slider",
                                help="Angekündigte Verweilzeit (etwa der Abholtermin, den der Spediteur meldet) = wahre Verweilzeit mal exp(ν z) mit zufälligem z. ν ist der Fehler der Ankündigung in Log-Einheiten: "
                                     "0,3 heißt typisch ±30 %. 'keine' lässt nur die Standardmerkmale. Der Regler ändert die wahren Verweilzeiten nicht, nur das Signal.")
    learner = st.selectbox("Lerner", list(C.LEARNERS), format_func=C.LEARNER_LABELS.get, key="learner_select",
                           help="Gilt für alle Prognose-Regeln und liefert Median, Mittelwert, Quantil 0,7 und Streuung der Verweilzeit. Das Gruppenmittel nach Art erscheint nur als Vergleichslinie in der Lernkurve.")
    seed = st.number_input("Zufalls-Seed (Block)", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt den Block: Ankünfte, Merkmale und Verweilzeiten.")
    train_seed = st.number_input("Trainings-Seed", *bounds("train_seed_input"), key="train_seed_input", step=1,
                                 help="Bestimmt die Trainingshistorie. Bei kleinen Trainingsmengen ist das Ergebnis eine Ziehung: ändern Sie den Seed und sehen Sie, wie es schwankt.")
    st.button("🎲 Neuer Block", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für den Block (Ankünfte, Merkmale, Verweilzeiten).")

sync_query_params({key: st.session_state[key] for key in SCENARIO_KEYS})

p = E.Params(int(n_stacks), int(max_height), int(fill_pct), int(n_containers), int(announce), int(train_idx), learner, int(train_seed))
view_key = ss.get("view_radio", C.VIEW_DEFAULT)
with st.spinner("Trainiere die Prognose und simuliere den Block..."):
    outcomes = _compute_block(p.key + (int(seed),))
with st.spinner("Rechne die Stichprobe (60 Blöcke)..."):
    sample = _compute_sample(p.key)
rows = sample.rows
by_key = {o.key: o for o in outcomes}
block = E.block_of(p, int(seed))
cap = block.inst.capacity

# ---------------------------------------------------------------------------------------------------
# Hauptansicht
# ---------------------------------------------------------------------------------------------------
st.markdown("## 🎯 Was bringt die gelernte Prognose gegenüber dem einfachen Ausgleich?")
n_train_text = f"{p.n_train:,}".replace(",", ".")
st.caption(f"Block {p.n_stacks} × {p.max_height} mit höchstens {cap} Containern gleichzeitig, {p.n_containers} Container, Ankündigung: {C.NU_LABELS[p.announce]}, {C.LEARNER_LABELS[p.learner]} mit "
           f"{n_train_text} Trainingscontainern. Angezeigt wird die gewählte Regel (siehe Blick in den Block); Delta = Regel minus Niedrigster Stapel ohne Prognose.")

chosen = by_key[view_key]
render_metrics(chosen, outcomes, p.n_containers)

diag = E.diagnose(view_key, rows, outcomes, p)
text = E.diagnosis_text(diag, view_key, int(seed))
if diag.kind == "helps":
    st.success(f"✅ {text}")
elif diag.kind == "tipped":
    st.warning(f"⚠️ {text}")
else:
    st.info(f"ℹ️ {text}")

st.markdown("#### 🔍 Blick in den Block")
view_key = st.radio("Rechts vergleichen mit", list(C.VIEW_KEYS), format_func=LABEL.get, key="view_radio", horizontal=True, help="Links steht immer der Bezug: Niedrigster Stapel ohne Prognose.")
right = by_key[view_key]
base_o = by_key[BASE]
scenario_key = (p, int(seed))
if st.session_state.get("event_owner") != scenario_key:
    st.session_state["event_slider"] = E.suggested_event(base_o.result)
    st.session_state["event_owner"] = scenario_key
event = event_control(block.inst.n_events)
est_time = block.t_arr + right.est_days                       # Rang nach der Prognose der rechten Regel (links wird sie nicht benutzt)
left_col, right_col = st.columns(2)
for col, outcome, side in ((left_col, base_o, "left"), (right_col, right, "right")):
    with col:
        stacks, moved, arrived, step = E.state_at(block.inst, outcome.result, event)
        so_far = step.moves_so_far if step is not None else 0
        st.plotly_chart(V.block_figure(stacks, p.max_height, block.t_dep, est_time, V.block_title(outcome.label, so_far), moved, arrived, block.dwell, right.est_days),
                        width="stretch", key=f"block_chart_{side}")
        st.caption(E.describe_step(step, event, block.inst.n_events))
st.caption(f"Links der Bezug, rechts {LABEL[view_key]}. {C.BLOCK_LEGEND_TEXT}")

pdf_slot = st.container()

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Kernabschnitt
# ---------------------------------------------------------------------------------------------------
st.subheader("📐 Wie gut muss die Prognose sein, und wie viele Daten braucht sie?")
st.markdown(
    """
Kernfrage dieser Demo: Ab wann schlägt eine gelernte Prognose den einfachen **Ausgleich** (Niedrigster Stapel, ganz ohne Prognose)? Eine Prognose ist nur so viel wert wie ihr **Signal**, und ihr Wert
hängt am Anteil **falsch geordneter Paare**, nicht am mittleren Fehler. Hier live für Ihre Einstellungen gerechnet, **mit der Verteilung dazu**:
"""
)
with st.spinner("Rechne Kurven (Gauß-Rauschen und Lernkurve)..."):
    gauss = _compute_gauss(p.block_key)
    lc = _compute_learning_curve(p.key)
tip = E.tipping_point(gauss)
mean_q70 = E.mean_of(rows, Q70)
q_q70 = sample.quality[Q70]
eq_sigma, eq_kind = E.equivalent_sigma(gauss, mean_q70)
k1, k2, k3 = st.columns(3)
k1.metric("Kipppunkt (Paare)", f"{tip.pair_err * 100:.0f} %" if tip.kind == "crosses" else "–",
          help=(f"Bestfit mit Gauß-Rauschen (Regler der Stapelplanung) schlägt den Ausgleich bis σ = {tip.sigma:.2f} Standzeiten, das sind {tip.pair_err * 100:.0f} % falsch geordnete Paare (MAE {tip.mae:.2f})."
                if tip.kind == "crosses" else "Im Bereich σ 0,1 bis 2,0 gibt es bei diesem Block keinen Kipppunkt: " + ("Bestfit gewinnt überall." if tip.kind == "always_better" else "Bestfit verliert überall.")))
k2.metric("Ihr Paarfehler", f"{q_q70.pair_err * 100:.0f} %",
          help=f"Bestfit vorsichtig (Quantil 0,7) mit dem eingestellten Lerner, gepoolt über die 60 Blöcke der Stichprobe. Der mittlere Fehler (MAE) liegt dabei bei {q_q70.mae:.2f} Standzeiten: "
               "er sagt weniger über den Nutzen als der Anteil falsch geordneter Paare.")
k3.metric("Wirkt wie σ", f"{'< ' if eq_kind == 'below' else '> ' if eq_kind == 'above' else ''}{eq_sigma:.2f}",
          help="Rauschen σ in mittleren Standzeiten: das Gauß-Rauschen der Stapelplanung, das auf denselben Blöcken dieselben Umstapelungen ergibt (auf der Kurve abgelesen). Die Größenordnung σ 25 bis 50 % der Stapelplanung entspricht einer guten Ankündigung.")

st.markdown("**Umstapelungen über dem Anteil falsch geordneter Paare**")
learned = [(n, pe, m) for n, pe, m in zip(lc.sizes, lc.pair_err, lc.moves)]
st.plotly_chart(V.pair_error_figure(gauss, tip, learned, (q_q70.pair_err, mean_q70), C.LEARNER_LABELS[p.learner]), width="stretch", key="pair_error_chart")
st.caption(f"Basis: {gauss.n_instances} Blöcke (Seeds 0-{gauss.n_instances - 1}, nicht Ihr Seed) mit Ihren Einstellungen. Grau das Gauß-Rauschen der Stapelplanung (σ 0,1 bis 2,0 Standzeiten), waagerecht der Ausgleich "
           f"({gauss.pure:.3f} je Container), als Raute der Hellseher ({gauss.hell:.3f}). Grün die gelernten Prognosen des eingestellten Lerners, beschriftet mit der Trainingsmenge, der graue Ring ist Ihre "
           "eingestellte Prognose. Wo die grüne Linie über dem Ausgleich liegt, verliert die Prognose. Gelernte Prognosen liegen etwa auf der Gauß-Kurve, wenn man sie über dem Paarfehler aufträgt, "
           "nicht über dem MAE.")

v = E.verdict(rows, Q70)
d = E.distribution(rows, Q70)
st.markdown("**Urteil über die Stichprobe** (gepaarte Differenz je Block, klar ab mehr als zwei Standardfehlern)")
if v.kind == "better":
    amount = f"**{abs(v.pct):.0f} % weniger**" if v.pct is not None else f"**{abs(v.diff):.3f} weniger**"
    st.success(f"✅ **Gelernt (Bestfit, Quantil 0,7) gegen Ausgleich**: im Mittel {amount} Umstapelungen ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}). An **{d.worse * 100:.0f} %** der Blöcke ist es umgekehrt.")
elif v.kind == "worse":
    amount = f"**{v.pct:.0f} % mehr**" if v.pct is not None else f"**{v.diff:.3f} mehr**"
    st.warning(f"⚠️ **Gelernt (Bestfit, Quantil 0,7) gegen Ausgleich**: im Mittel {amount} Umstapelungen ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}). An **{d.better * 100:.0f} %** der Blöcke ist es besser.")
else:
    st.info(f"ℹ️ Kein klarer Unterschied bei **gelernt (Bestfit, Quantil 0,7) gegen Ausgleich**: die Differenz ({v.diff:+.3f} je Container, gemittelt über die Blöcke) liegt innerhalb des Rauschens "
            f"(Standardfehler {v.se:.3f}). Besser an {d.better * 100:.0f} %, schlechter an {d.worse * 100:.0f} % der Blöcke.")
st.plotly_chart(V.distribution_figure(E.diffs(rows, Q70), v.diff, "gelernt (Quantil 0,7)"), width="stretch", key="distribution_chart")
st.caption(f"Basis: {len(rows)} Blöcke (Seeds 0-{len(rows) - 1}, nicht Ihr Seed). Besser an {d.better * 100:.0f} %, gleich an {d.equal * 100:.0f} %, schlechter an {d.worse * 100:.0f} % der Blöcke; Median der Differenz "
           f"{d.median:+.3f} je Container." + (f" Die besten 10 % der Blöcke tragen {d.top_decile_share * 100:.0f} % der gesamten Ersparnis." if d.top_decile_share is not None else ""))

st.markdown("**Wie viele Trainingsdaten braucht die Prognose?**")
hell_diff = E.mean_of(rows, HELL) - E.mean_of(rows, BASE)
st.plotly_chart(V.learning_curve_figure(lc, hell_diff, p.learner, p.n_train), width="stretch", key="learning_curve_chart")
st.caption(f"Basis: {lc.n_instances} Blöcke (Seeds 0-{lc.n_instances - 1}); Differenz zum Ausgleich je Container, negativ ist besser. Das Band zeigt bei weniger als {C.LEARNING_CURVE_BAND_BELOW} Trainingscontainern "
           f"das Ergebnis über {len(lc.seeds)} Trainings-Seeds (ab Ihrem Seed {lc.seeds[0]}): bei kleiner Trainingsmenge ist das Ergebnis eine Ziehung. Gestrichelt der Ausgleich und der Hellseher; "
           "das Gruppenmittel nach Art (Mittelwert je Containerart) ignoriert Ankündigung und Merkmale bis auf die Art und schlägt den Ausgleich nie.")

st.markdown("**Median oder Quantil? Der beste mittlere Fehler ist nicht das wenigste Umstapeln**")
bars = [("Median", E.mean_of(rows, MED), sample.quality[MED].mae, C.RULE_COLORS[MED]), ("Mittelwert", E.mean_of(rows, MEAN), sample.quality[MEAN].mae, C.RULE_COLORS[MEAN]),
        ("Quantil 0,7", mean_q70, q_q70.mae, C.RULE_COLORS[Q70])]
st.plotly_chart(V.forecast_kind_figure(bars, E.mean_of(rows, BASE)), width="stretch", key="forecast_kind_chart")
st.caption(f"Basis: {len(rows)} Blöcke, derselbe Lerner ({C.LEARNER_LABELS[p.learner]}). Der Median hat den kleinsten mittleren Fehler (MAE), aber nicht das wenigste Umstapeln: die Regel vergleicht Abfahrtszeiten "
           "mehrerer Container, und große Ausreißer der schiefen Verweilzeit schaden mehr, als der MAE zählt. In der Messreihe gewinnt das vorsichtige Quantil bei einem Füllgrad ab 80 %, bei 60 % gibt es "
           "keinen Unterschied.")

with pdf_slot:
    st.download_button(
        "📄 Ergebnis als PDF herunterladen",
        data=generate_vwz_pdf(p, int(seed), view_key, outcomes, diag, rows=rows, sample=sample, gauss=gauss, lc=lc),
        file_name="verweilzeit_ergebnis.pdf", mime="application/pdf", key="primary_pdf_download",
        help="Szenario, Regelvergleich, Meldung, Stichprobe mit Urteil, Kipppunkt und Lernkurve.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Regelvergleich
# ---------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – Regeln im Vergleich"):
    tabs = st.tabs([LABEL[k] for k in C.TAB_RULES] + ["📊 Vergleich"])
    for tab, key in zip(tabs, C.TAB_RULES):
        with tab:
            render_rule_panel(f"rule_{key}", by_key[key], outcomes, p.n_containers, dwell=block.dwell)
    with tabs[len(C.TAB_RULES)]:
        table = []
        for o in outcomes:
            q = sample.quality[o.key]
            is_base = o.key == BASE
            table.append({"Regel": C.RULE_SHORT[o.key], "Umstapelungen je Container (Block)": round(o.moves / p.n_containers, 3),
                          "Delta gegen Ausgleich (Block)": None if is_base else round((o.moves - by_key[BASE].moves) / p.n_containers, 3),
                          "Delta gegen Ausgleich (Stichprobe ± Standardfehler)": "–" if is_base else f"{E.paired(rows, o.key)[0]:+.3f} ± {E.paired(rows, o.key)[1]:.3f}",
                          "MAE (Stichprobe)": None if q is None else round(q.mae, 2), "Falsch geordnete Paare % (Stichprobe)": None if q is None else round(q.pair_err * 100)})
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
        st.plotly_chart(V.rule_bar_figure(outcomes, p.n_containers), width="stretch", key="comparison_bar_chart")
        st.caption("Ein Block, sieben Regeln; Delta immer Regel minus Niedrigster Stapel ohne Prognose. Die Zeile 'Niedr. Stapel + Prognose' legt den Ausgleich beim Einlagern, benutzt die Prognose (Quantil 0,7) aber "
                   "beim Umstapeln, wie die Stapelplanung es für alle Regeln tut: sie zeigt, dass schon das Umstapeln mit Prognose einen Teil des Nutzens bringt. Der Hellseher kennt die wahre Abfahrt, "
                   "nicht künftige Ankünfte (anders als das Optimum der Stapelplanung). Unsicherheitsbewusst und Median benutzen dieselbe Prognose (Median).")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        """
**Der synthetische Strom.** Container kommen als Poisson-Strom an einem Block an; die Belegungsgrenze folgt der Stapelplanung (ist der Block voll, verzögert sich die Ankunft bis zur nächsten Abfahrt).
Jeder Container hat **Merkmale, die bei Ankunft bekannt sind**: Containerart (Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %), Kunde (12 Kunden mit einem Effekt), Zollhalt (nur Import), Wochenende der
Ankunft, Vorlauf bis zur Schiffsabfahrt (Export, Umschlag) und ein reines Rauschmerkmal. Die **Verweilzeit** wird danach gezogen: lognormal und **heteroskedastisch** (Export streut wenig, Leer viel), im
Mittel rund 5,6 Tage. Dazu gibt es optional die **Ankündigung**: die wahre Verweilzeit mal exp(ν z), etwa der Abholtermin, den der Spediteur meldet. Alles ist ausdrücklich **erfunden**: Merkmale, Verteilung
und Größenordnungen sind Annahmen, keine Messung an einem Terminal.

**Lernen.** Der Lerner sieht eine unabhängige Historie mit derselben Verteilung (abgeschlossene Container; kein Container des Blocks steckt darin, und die Merkmale enthalten keine Zukunftsinformation).
- **Lineare Regression** auf die log-Verweilzeit mit Art- und Kunden-Dummies, Zoll, Wochenende, Vorlauf, Rauschmerkmal und der log-Ankündigung; die Streuung ist eine Konstante.
- **Gradient Boosting** (selbst gebaut, nur numpy): Histogramm-Bäume der Tiefe 3, 100 Runden, Lernrate 0,1, mit Quantilverlust (0,1 / 0,5 / 0,7 / 0,9 auf der log-Verweilzeit) oder L2; die Streuung folgt aus den
  Quantilen 0,1 und 0,9, sie ist also je Container verschieden. Nicht abgestimmt.
- Das **Gruppenmittel nach Art** erscheint nur als Vergleichslinie in der Lernkurve.

**Drei Prognosen.** Jeder Lerner liefert den **Median** (die MAE-optimale Prognose), den **Mittelwert** und ein vorsichtiges **Quantil 0,7** der Verweilzeit. Die Schätzung der Abfahrt ist Ankunftszeit plus
Prognose. Der Median stellt die falsche Frage: die Regel vergleicht die Abfahrtszeiten **mehrerer** Container, und bei der schiefen Verweilzeit schaden große Ausreißer mehr, als der mittlere Fehler zählt.

**Regeln.** Alle laufen auf derselben Ereignisfolge (dieselben Merkmale und Zufallszahlen) und sind die der Stapelplanung, unverändert.
- **🚛 Niedrigster Stapel** (Bezug): Stapel mit den wenigsten Containern, auch beim Umstapeln; ganz ohne Prognose.
- **🎯 Bestfit mit Median**, **🛡️ Bestfit vorsichtig (Quantil 0,7)**: Bestfit auf der geschätzten Abfahrt; beim Umstapeln entscheidet ebenfalls Bestfit.
- **🎲 Unsicherheitsbewusst**: wählt den Stapel mit der kleinsten Wahrscheinlichkeit, einen früher abfahrenden Container zu blockieren, aus den prognostizierten Verteilungen.
- Im Vergleich zusätzlich Bestfit mit Mittelwert, der Hellseher (wahre Abfahrt) und der Ausgleich beim Einlagern mit Umstapeln nach der Prognose.

**Bezug.** Jede Ersparnis ist gegen **Niedrigster Stapel ohne jede Prognose** gemessen: das ist die ehrliche Antwort auf die Frage, was die Prognose bringt. Die Stapelplanung lässt alle Regeln mit derselben
(verrauschten) Prognose umstapeln; auch diese Variante steht als eigene Zeile im Vergleich.

**Warum der Paarfehler das Maß ist.** Umstapeln entsteht, wenn ein später abfahrender Container über einem früher abfahrenden liegt: es kommt auf die **Reihenfolge** an, nicht auf den Betrag des Fehlers. Der
Nutzen hängt darum am Anteil falsch geordneter Paare unter den gleichzeitig anwesenden Containern. Das Gauß-Rauschen der Stapelplanung (ein Regler dort) liegt auf derselben Kurve; wo sie den Ausgleich
kreuzt, liegt der Kipppunkt.

**Kleine Trainingsmengen sind ein Glücksspiel, und warum 300 Container.** Mit wenigen Trainingscontainern hängt das Ergebnis an der Ziehung der Historie (Trainings-Seed) und kann den Ausgleich verfehlen; die
Lernkurve zeigt das Band über mehrere Seeds. Auch der einzelne Block schwankt mit der Historie, deshalb tragen die Preset-Geschichten an der Grundgesamtheit über 200 Blöcke und nicht nur am gezeigten Block.
Die Zahlen hängen an der Lauflänge: bei 120 Containern sind Anlauf und Auslauf ein großer Teil des Laufs, deshalb ist der Standard 300.

**Grenzen dieses Modells** (bewusst so gewählt, damit die Aussage ehrlich bleibt):
- Merkmale, Verweilzeit-Verteilung und Ankündigung (mit lognormalem Fehler, ohne Ausreißer und fehlende Werte) sind **Annahmen**; ein realer Betreiber hätte andere Merkmale.
- **Die unerklärte Streuung ist die entscheidende Annahme.** Messreihe (120 Container, 200 Blöcke; Differenz zum Ausgleich): wird die Streuung der Verweilzeit halbiert, spart schon die exakte Kenntnis der
  Standardmerkmale −0,272 je Container, bei 1,5-facher Streuung kostet sie +0,180. Ob sich Lernen aus Standardmerkmalen lohnt, hängt also vollständig an der angenommenen Unvorhersagbarkeit.
- Training und Test stammen aus derselben Verteilung (kein Saisonwechsel, kein Bruch); in der Messreihe kostet eine um 30 % zu kurze Prognose rund die Hälfte des Nutzens, eine zu lange schadet weniger.
- Die Abholreihenfolge folgt allein der Verweilzeit (keine Bündelung nach Schiff); ein Block, keine Kranwege; der Hellseher kennt keine künftigen Ankünfte.
- Was eine bessere Prognose kostet (Daten, Erhebung, Pflege), ist Sache des Betreibers und **kein Geldbetrag** der Demo. Alle Zahlen sind **Größenordnungen aus einer Simulation, keine Messung an einem Terminal**.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Verweilzeit.** Zu Merkmalen $x$ ist $D \mid x \sim \mathrm{LogNormal}(\mu(x),\, s(x)^2)$, die Abfahrt $T = t_{\mathrm{an}} + D$. Prognose der Verweilzeit: $q = e^{\mu}$ (Median),
$q = e^{\mu + s^2/2}$ (Mittelwert) oder $q = e^{\mu + 0{,}524\, s}$ (Quantil 0,7, denn $\Phi^{-1}(0{,}7) = 0{,}524$). Die Schätzung der Abfahrt ist $\hat e = t_{\mathrm{an}} + q$.

**Ankündigung.** $A = D\, e^{\nu z}$ mit $z \sim \mathcal N(0,1)$ unabhängig. Die exakte bedingte Verteilung von $\ln D$ gegeben Merkmale und $A$ ist wieder normal mit Präzisionsgewicht
$\mu' = \dfrac{\mu / s^2 + \ln A / \nu^2}{1/s^2 + 1/\nu^2}$ und $s'^2 = \dfrac{1}{1/s^2 + 1/\nu^2}$ (das Bayes-Orakel, Obergrenze jedes Lerners; in der Demo im Test geprüft).

**Lerner.** Pinball-Verlust $\rho_\alpha(u) = \max(\alpha u,\ (\alpha - 1) u)$. *Lineare Regression:* OLS auf $\ln D$, Streuung $\hat s^2 = \mathrm{RSS} / (n - p)$. *Boosting:* additives Modell aus Bäumen der Tiefe 3,
Startwert = $\alpha$-Quantil (bei L2 der Mittelwert), je Runde ein Baum auf dem Gradienten des Verlusts, Blattwert = $\alpha$-Quantil der Residuen im Blatt (bei L2 der Mittelwert), Lernrate 0,1. Die
Streuung des Boostings ist $\max\{(q_{0,9} - q_{0,1})/2{,}563,\ 0{,}05\}$ auf der log-Skala.

**Regeln.** Bestfit: der Container geht auf den Stapel, dessen oberster nach $\hat e$ am knappsten nicht früher abfährt ($\hat e_{\mathrm{oben}} \ge \hat e_{\mathrm{eigen}}$, kleinstes $\hat e_{\mathrm{oben}}$); blockiert jeder Stapel,
auf den mit dem größten $\hat e_{\mathrm{oben}}$. Niedrigster Stapel: wenigste Container, kleinster Index. Unsicherheitsbewusst: $P(T_{\mathrm{oben}} < T_{\mathrm{eigen}}) = \mathbb{E}_z\big[F_{\mathrm{oben}}(t_{\mathrm{eigen}} + e^{\mu + s z})\big]$
per Gauß-Hermite mit 12 Knoten, auf zwei Stellen gerundet, unterhalb von 0,3 gilt die knappste Passung, sonst der niedrigste Stapel.

**Kennzahlen.** Umstapelungen je Container $= (\text{Hübe}) / n$. Prognosefehler MAE $= \overline{|q - D|} / \bar D$ in mittleren Standzeiten. **Paarfehler** $=$ Anteil der Paare $(i, j)$ gleichzeitig anwesender
Container ($t_i^{\mathrm{an}} < T_j$ und $t_j^{\mathrm{an}} < T_i$) mit $\operatorname{sign}(\hat e_i - \hat e_j) \ne \operatorname{sign}(T_i - T_j)$. Gauß-Rauschen der Stapelplanung: $\hat e = T + \sigma \bar D\, \varepsilon$ mit $\varepsilon \sim \mathcal N(0,1)$.

**Vergleich über Blöcke.** Für Regel $A$ gegen den Bezug $B$ auf denselben Blöcken $d = 1, \dots, 60$ ist $\Delta_d = X_A^{(d)} - X_B^{(d)}$ die Differenz der Umstapelungen je Container (negativ = besser). Ein Unterschied gilt als
klar, wenn $|\bar\Delta| > 2\,\mathrm{SE}(\Delta)$ mit dem Standardfehler der gepaarten Differenz, sonst wird kein Urteil gefällt. **Kipppunkt:** erster Vorzeichenwechsel von (Bestfit mit Rauschen $\sigma$) minus Ausgleich
auf dem Raster $\sigma = 0{,}1 \dots 2{,}0$, linear interpoliert; **wirkt wie Rauschen** $\sigma$: die Umstapelungen der Prognose auf derselben Kurve abgelesen.

Implementiert in `vwz_generator.py` (Strom), `vwz_models.py` (Lerner), `vwz_forecast.py` (Prognose, Schätzung, unsicherheitsbewusste Regel), `vwz_stk_core.py` (Kopie der Stapelplanung),
`vwz_evaluation.py` (Regeln, Stichprobe, Kurven, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
