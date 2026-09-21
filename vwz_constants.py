"""Konstanten der Verweilzeit-Demo: Modell (synthetischer Strom), Lerner, Regeln, Auswertung, Darstellung, Regler und Presets.

Die Modellparameter sind die der Messreihe (hafen-planung/messreihe_verweilzeit), damit die dort gemessenen Zahlen mit diesem Code reproduzierbar sind. Alle Merkmale, die Verweilzeit-Verteilung
und die Größenordnungen sind ERFUNDEN (synthetisch, kein Echtdatenbezug)."""

from statistics import NormalDist

# --- Modell: Merkmale und Verweilzeit --------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)            # Anteile der Containerarten
K_KUNDEN = 12                                # Kunden mit je einem Effekt auf die log-Verweilzeit
KUNDEN_SEED = 12345
U_SD = 0.30                                  # Streuung der Kundeneffekte (log)
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art (bei Export und Umschlag ein Drittel)
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art (heteroskedastisch)
P_ZOLL = 0.25                                # Zollhalt bei Import
FEATURES = ("art", "kunde", "zoll", "wochenende", "vorlauf", "gewicht", "angekuendigt")
N_FEATURES = len(FEATURES)
COL_ANNOUNCED = FEATURES.index("angekuendigt")
RHO = 0.7                                    # Ankunftslast: mittlere Belegung wie in der Stapelplanung
HISTORY_SEED_OFFSET = 10_000_000             # Trainingshistorien haben einen eigenen Zufallsstrom-Bereich
MEAN_DWELL_SEED, MEAN_DWELL_N = 999, 400_000  # Stichprobe für die mittlere Verweilzeit (legt die Ankunftsrate fest)

# --- Ankündigung: Verlässlichkeit nu (Fehler der Ankündigung in Log-Einheiten); Index 0 = keine Ankündigung ---
NU_LEVELS = (None, 1.0, 0.6, 0.3, 0.15)
NU_LABELS = ("keine", "unzuverlässig (ν 1,0)", "mittel (ν 0,6)", "gut (ν 0,3)", "sehr gut (ν 0,15)")
NU_SHORT = ("keine Ankündigung", "ν 1,0", "ν 0,6", "ν 0,3", "ν 0,15")

# --- Lerner -----------------------------------------------------------------------------------------------
LEARNER_LINEAR, LEARNER_BOOSTING, LEARNER_GROUP = "linear", "boosting", "gruppe"
LEARNERS = (LEARNER_LINEAR, LEARNER_BOOSTING)              # wählbar; das Gruppenmittel erscheint nur als Vergleich in der Lernkurve (die Ankündigung wäre dort wirkungslos)
LEARNER_LABELS = {LEARNER_LINEAR: "Lineare Regression", LEARNER_BOOSTING: "Gradient Boosting", LEARNER_GROUP: "Gruppenmittel nach Art"}
TRAIN_SIZES = (30, 100, 300, 1000, 3000, 10000)             # Trainingscontainer
BOOST_ROUNDS, BOOST_DEPTH, BOOST_LR, BOOST_BINS, BOOST_MIN_LEAF = 100, 3, 0.1, 64, 5
QUANTILE = 0.7                                              # vorsichtiges Quantil (fest)
Z_QUANTILE = NormalDist().inv_cdf(QUANTILE)                 # 0,5244
S_FLOOR = 0.05                                              # untere Grenze der Log-Streuung des Boostings
S_BAND_DIVISOR = 2.563                                      # (q90 - q10) / 2,563 = Streuung einer Normalverteilung
FEW_DATA_BELOW = 300                                        # unter so vielen Trainingscontainern gilt 'zu wenig Daten'

# --- Regeln --------------------------------------------------------------------------------------------------
RULE_LOWEST = "niedrigster_stapel"          # Bezug: Einlagern und Umstapeln nach niedrigstem Stapel, ganz ohne Prognose
RULE_MEDIAN = "bestfit_median"
RULE_Q70 = "bestfit_q70"
RULE_AWARE = "unsicherheitsbewusst"
RULE_MEAN = "bestfit_mittelwert"
RULE_LOWEST_SAME = "niedrigster_stapel_prognose"
RULE_HELL = "hellseher"
RULE_KEYS = (RULE_LOWEST, RULE_MEDIAN, RULE_Q70, RULE_AWARE, RULE_MEAN, RULE_LOWEST_SAME, RULE_HELL)
TAB_RULES = (RULE_LOWEST, RULE_MEDIAN, RULE_Q70, RULE_AWARE)      # je Regel ein Tab; Mittelwert, Hellseher und 'gleiche Prognose' stehen nur im Vergleich
VIEW_KEYS = (RULE_MEDIAN, RULE_Q70, RULE_AWARE)                   # wählbar im Blick in den Block und in den Kennzahlen (rechts); links steht immer der Bezug
VIEW_DEFAULT = RULE_Q70
BASELINE = RULE_LOWEST
RULE_LABELS = {
    RULE_LOWEST: "🚛 Niedrigster Stapel",
    RULE_MEDIAN: "🎯 Bestfit mit Median",
    RULE_Q70: "🛡️ Bestfit vorsichtig (Quantil 0,7)",
    RULE_AWARE: "🎲 Unsicherheitsbewusst",
    RULE_MEAN: "Bestfit mit Mittelwert",
    RULE_LOWEST_SAME: "Niedrigster Stapel, Umstapeln mit Prognose",
    RULE_HELL: "🔮 Hellseher (Obergrenze)",
}
RULE_SHORT = {RULE_LOWEST: "Niedrigster Stapel", RULE_MEDIAN: "Bestfit Median", RULE_Q70: "Bestfit Quantil 0,7", RULE_AWARE: "Unsicherheitsbewusst", RULE_MEAN: "Bestfit Mittelwert",
              RULE_LOWEST_SAME: "Niedr. Stapel + Prognose", RULE_HELL: "Hellseher"}
RULE_COLORS = {RULE_LOWEST: "#2a6fb0", RULE_MEDIAN: "#c77700", RULE_Q70: "#2e7d4f", RULE_AWARE: "#7a3fb0", RULE_MEAN: "#a8508a", RULE_LOWEST_SAME: "#7fa8d4", RULE_HELL: "#5b6470"}
RULE_DESCRIPTIONS = {
    RULE_LOWEST: "Nimmt den Stapel mit den wenigsten Containern, auch beim Umstapeln, und benutzt keine Prognose. Das ist der **Bezug aller Vergleiche**: der Alltag ohne Prognose.",
    RULE_MEDIAN: "Bestfit auf Ankunft plus **Median** der prognostizierten Verweilzeit: der Container geht dorthin, wo oben der nächstspäter abfahrende liegt. Der Median ist die naheliegende Prognose "
                 "und im Mittel der Beträge (MAE) die beste.",
    RULE_Q70: "Bestfit auf Ankunft plus **Quantil 0,7** der prognostizierten Verweilzeit (je Container ein eigener Aufschlag). Der MAE ist schlechter als beim Median, das Umstapeln aber oft besser: "
              "die Regel vergleicht Abfahrtszeiten mehrerer Container, und große Ausreißer schaden mehr, als der MAE zählt.",
    RULE_AWARE: "Wählt den Stapel mit der kleinsten Wahrscheinlichkeit, einen früher abfahrenden Container zu blockieren, aus den prognostizierten Verteilungen (Median und Streuung je Container). "
                "Regel der Stapelplanung, hier unverändert übernommen; sie hilft meist nicht.",
    RULE_MEAN: "Bestfit auf Ankunft plus **Mittelwert** der prognostizierten Verweilzeit (Boosting auf log-Verweilzeit mit Aufschlag halbe Varianz).",
    RULE_LOWEST_SAME: "Einlagern nach niedrigstem Stapel, Umstapeln aber mit der Prognose (Quantil 0,7), wie die Stapelplanung es für alle Regeln tut. Isoliert die Einlagerungsentscheidung.",
    RULE_HELL: "Bestfit mit der **wahren** Abfahrt. Kennt keine künftigen Ankünfte (anders als das Optimum der Stapelplanung); eine Obergrenze für jede Prognose.",
}
# Bei Gleichstand der Umstapelungen gewinnt die Regel ohne Prognose (kein Scheinvorteil an Gleichständen).

# --- Auswertung --------------------------------------------------------------------------------------------
GAUSS_SIGMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)   # Rauschen der Stapelplanung in Standzeiten
GAUSS_NOISE_SEED = (13, 5)                                     # Rauschseed je Instanz i: 13 * i + 5
SAMPLE_INSTANCES = 60                                          # Blöcke der Stichprobe (Seeds 0 bis 59, bewusst nicht der eingestellte Seed)
LEARNING_CURVE_EXTRA_SEEDS = 3                                 # weitere Trainings-Seeds für das Band bei kleinen Trainingsmengen
LEARNING_CURVE_BAND_BELOW = 1000                               # Band nur unterhalb dieser Trainingsmenge
VERDICT_Z = 2.0                                                # klar = mehr als so viele Standardfehler der gepaarten Differenz
STATIC_TOL = 1e-9

# --- Darstellung -----------------------------------------------------------------------------------------------
BLOCK_COLOR_SOON = (31, 58, 95)
BLOCK_COLOR_LATE = (208, 224, 240)
BLOCK_STACK_BG = "#f0f2f5"
BLOCK_STACK_LINE = "#c9d1db"
MOVED_COLOR = "#e8850c"
ARRIVED_COLOR = "#2e7d4f"
MARKER_LINE_COLOR = "#808895"              # mittleres Grau: sichtbar auf hellem UND dunklem Hintergrund
GAUSS_COLOR = "#8a94a3"
BLOCK_FIGURE_TIER_PX = 38
BLOCK_FIGURE_BASE_PX = 70
CHART_HEIGHT = 380
BLOCK_LEGEND_TEXT = ("Farbe = wahre Abfahrtsreihenfolge (dunkel = fährt als nächster ab), Zahl = Rang nach der Prognose der rechten Regel (links wird sie nicht benutzt; beim Hellseher "
                     "gleich der wahren), oranger Rand = gerade umgestapelt, grüner Rand = gerade angekommen.")

# --- Regler (Bereich als (min, max), Standardwert, Schrittweite) -------------------------------------------------
N_STACKS_RANGE, N_STACKS_DEFAULT = (3, 8), 6
MAX_HEIGHT_RANGE, MAX_HEIGHT_DEFAULT = (3, 6), 5
FILL_PCT_RANGE, FILL_PCT_DEFAULT, FILL_PCT_STEP = (40, 100), 80, 5
N_CONTAINERS_RANGE, N_CONTAINERS_DEFAULT, N_CONTAINERS_STEP = (100, 500), 300, 50
ANNOUNCE_RANGE, ANNOUNCE_DEFAULT = (0, len(NU_LEVELS) - 1), 2
TRAIN_RANGE, TRAIN_DEFAULT = (0, len(TRAIN_SIZES) - 1), 3
LEARNER_DEFAULT = LEARNER_BOOSTING
SEED_RANGE, SEED_DEFAULT = (0, 9999), 268
TRAIN_SEED_RANGE, TRAIN_SEED_DEFAULT = (0, 99), 5

# Beispielszenarien (Schnellstart). Alle: Block 6 x 5, 300 Container, Boosting, gemeinsamer Block-Seed 268 (aus den Seeds 100 bis 299 gewählt, siehe tools/PRESET_SWEEP.md), Trainings-Seed 5.
_BASE = dict(n_stacks=6, max_height=5, fill_pct=80, n_containers=300, learner=LEARNER_BOOSTING, seed=268, train_seed=5)
PRESETS = {
    "Nur Standard": dict(_BASE, announce=0, train_idx=3),
    "Mit Ankündigung": dict(_BASE, announce=2, train_idx=3),
    "Zu wenig Daten": dict(_BASE, announce=2, train_idx=0),
    "Sehr verlässlich": dict(_BASE, announce=3, train_idx=3),
    "Voller Block": dict(_BASE, announce=2, train_idx=3, fill_pct=100),
}
