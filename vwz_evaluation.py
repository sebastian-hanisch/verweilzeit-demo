"""Auswertung: Regeln auf einem Block, Stichprobe über viele Blöcke, Gauß-Kurve der Stapelplanung, Lernkurve, Paarfehler, gepaarte Differenz, Urteil, Kipppunkt, äquivalentes Rauschen, Meldung.

Alles hier ist reine Rechnung ohne Streamlit; app.py legt die teuren Teile per st.cache_data ab. Die Rechnung folgt `hafen-planung/messreihe_verweilzeit` (sweeplib.py, lab.py).
Kennzahlen: Umstapelungen je Container (Nutzen) getrennt von der Prognosegüte (MAE in Standzeiten, Anteil falsch geordneter Paare); Delta immer 'Regel minus Bezug (Niedrigster Stapel ohne Prognose)'."""

import statistics
from dataclasses import astuple, dataclass
from functools import lru_cache

import numpy as np

import vwz_constants as C
import vwz_forecast as F
import vwz_generator as G
import vwz_stk_core as K


# ---------------------------------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    n_stacks: int
    max_height: int
    fill_pct: int
    n_containers: int
    announce: int            # Index in NU_LEVELS
    train_idx: int           # Index in TRAIN_SIZES
    learner: str
    train_seed: int

    @property
    def key(self):
        """Alle Einstellungen als Tupel (Schlüssel für die Zwischenspeicher der App)."""
        return astuple(self)

    @property
    def nu(self):
        return C.NU_LEVELS[self.announce]

    @property
    def n_train(self):
        return C.TRAIN_SIZES[self.train_idx]

    @property
    def fill(self):
        return self.fill_pct / 100

    @property
    def block_key(self):
        """Alles, was den Block (Ereignisfolge, Zeiten) festlegt, ohne Ankündigung und Lerner."""
        return (self.n_stacks, self.max_height, self.fill_pct, self.n_containers)


def params_from_preset(preset):
    return Params(preset["n_stacks"], preset["max_height"], preset["fill_pct"], preset["n_containers"], preset["announce"], preset["train_idx"], preset["learner"], preset["train_seed"])


@lru_cache(maxsize=192)
def make_block(n_stacks, max_height, fill_pct, n_containers, nu, seed):
    """Ein Block (Stichprobe des Generators); rein deterministisch, im Prozess gemerkt."""
    return G.generate(n_stacks, max_height, fill_pct / 100, n_containers, seed, nu=nu)


def block_of(p, seed):
    return make_block(p.n_stacks, p.max_height, p.fill_pct, p.n_containers, p.nu, seed)


# ---------------------------------------------------------------------------------------------------
# Prognosegüte
# ---------------------------------------------------------------------------------------------------
def pair_counts(t_arr, t_dep, est):
    """(falsch geordnete Paare, Paare) unter gleichzeitig anwesenden Containern: Paar (i, j) zählt, wenn sich ihre Aufenthalte überlappen; falsch, wenn die geschätzte Abfahrtsreihenfolge
    von der wahren abweicht."""
    n = len(t_dep)
    ii, jj = np.triu_indices(n, 1)
    present = (t_arr[ii] < t_dep[jj]) & (t_arr[jj] < t_dep[ii])
    a, b = ii[present], jj[present]
    wrong = int(np.sum(np.sign(est[a] - est[b]) != np.sign(t_dep[a] - t_dep[b])))
    return wrong, len(a)


def mae_days(pred, dwell):
    return float(np.mean(np.abs(pred - dwell)))


@dataclass(frozen=True)
class Quality:
    mae: float          # mittlerer Betragsfehler der Verweilzeit in Standzeiten (mittlere Verweilzeit = 1)
    pair_err: float     # Anteil falsch geordneter Abfahrten unter gleichzeitig anwesenden Paaren


def quality_est(samples, ests, d0=None):
    """Güte einer Schätzung der Abfahrtszeit über Blöcke (gepoolt): MAE der Abfahrt / mittlere Verweilzeit d0 (Standard: Mittel der Blockmittel), Paarfehler über alle Blöcke."""
    d0 = float(np.mean([s.mean_dwell_days for s in samples])) if d0 is None else d0
    err = np.concatenate([np.abs(e - s.t_dep) for s, e in zip(samples, ests)])
    wrong = tot = 0
    for s, e in zip(samples, ests):
        w, t = pair_counts(s.t_arr, s.t_dep, e)
        wrong += w
        tot += t
    return Quality(float(err.mean() / d0), wrong / tot if tot else 0.0)


def quality_of(samples, preds, d0=None):
    """Güte einer Prognose der Verweilzeit (Tage je Container) über Blöcke: die Schätzung der Abfahrt ist Ankunft + Prognose."""
    return quality_est(samples, [s.t_arr + p for s, p in zip(samples, preds)], d0)


def _clip(pred):
    return np.maximum(pred, 1e-3)


# ---------------------------------------------------------------------------------------------------
# Regeln auf einem Block
# ---------------------------------------------------------------------------------------------------
def pure_moves(sample):
    """Bezug: Einlagern UND Umstapeln nach niedrigstem Stapel, ganz ohne Prognose (die Schätzung wird nicht gelesen)."""
    return K.run(sample.inst, F.true_est(sample), K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel).moves


def rule_moves(sample, key, fc, record=False):
    """Umstapelungen der Regel `key` auf dem Block mit der Prognose `fc` (vwz_forecast.Forecast; für Hellseher und Bezug None)."""
    inst = sample.inst
    if key == C.RULE_LOWEST:
        return K.run(inst, F.true_est(sample), K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel, record=record)
    if key == C.RULE_HELL:
        return K.run(inst, F.true_est(sample), K.bestfit, record=record)
    if key == C.RULE_AWARE:
        return K.run(inst, F.make_est(sample.t_arr, fc.mu, fc.s), F.make_aware(), record=record)
    est = {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.q70, C.RULE_MEAN: fc.mean, C.RULE_LOWEST_SAME: fc.q70}[key]
    place = K.niedrigster_stapel if key == C.RULE_LOWEST_SAME else K.bestfit
    return K.run(inst, F.est_plain(sample.t_arr + _clip(est)), place, record=record)


def forecast_for(p, samples):
    """Prognose des eingestellten Lerners für alle Container der Blöcke (in einem Zug); zerlegt in ein Forecast je Block."""
    X = np.vstack([s.X for s in samples])
    fc = F.forecast(p.learner, p.nu, p.n_train, p.train_seed, X)
    off = np.cumsum([0] + [len(s.X) for s in samples])
    return [F.Forecast(*(getattr(fc, f)[off[i]:off[i + 1]] for f in ("median", "mean", "q70", "mu", "s"))) for i in range(len(samples))]


def forecast_of(rule_key, fc):
    """Prognose der Verweilzeit (Tage), die die Regel benutzt, für Güte und Streudiagramm; None für Bezug und Hellseher."""
    return {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.q70, C.RULE_MEAN: fc.mean, C.RULE_AWARE: fc.median, C.RULE_LOWEST_SAME: fc.q70}.get(rule_key)


@dataclass(frozen=True)
class Outcome:
    key: str
    label: str
    result: object                # vwz_stk_core.Result
    quality: object               # Quality oder None (Bezug ohne Prognose); Hellseher: 0 und 0
    est_days: object              # geschätzte Verweilzeit je Container (Tage) oder None

    @property
    def moves(self):
        return self.result.moves


def run_rules(p, seed, record=True):
    """Alle sieben Regeln auf demselben Block (Seed), mit derselben Prognose; Reihenfolge C.RULE_KEYS."""
    sample = block_of(p, seed)
    fc = forecast_for(p, [sample])[0]
    out = []
    for key in C.RULE_KEYS:
        pred = forecast_of(key, fc)
        if key == C.RULE_HELL:
            q = Quality(0.0, 0.0)
        elif pred is None:
            q = None
        else:
            q = quality_of([sample], [_clip(pred)])
        out.append(Outcome(key, C.RULE_LABELS[key], rule_moves(sample, key, fc, record), q, None if pred is None else _clip(pred)))
    return tuple(out)


def outcome_of(outcomes, key):
    return next(o for o in outcomes if o.key == key)


# ---------------------------------------------------------------------------------------------------
# Stichprobe über viele Blöcke
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    """Ein Block der Stichprobe: Umstapelungen je Container je Regel."""
    seed: int
    m: dict


@dataclass(frozen=True)
class Sample:
    rows: tuple
    quality: dict          # Regel -> Quality (Bezug: None)
    n_containers: int

    def __len__(self):
        return len(self.rows)


def sample(p, n=C.SAMPLE_INSTANCES, seed0=0, rules=C.RULE_KEYS, with_quality=True):
    """Stichprobe: Blöcke mit den Seeds seed0 .. seed0+n-1 (bewusst NICHT der eingestellte Seed), alle Regeln auf denselben Blöcken und derselben Prognose. with_quality=False lässt MAE und Paarfehler
    aus (teuer bei vielen Blöcken; die Grundgesamtheit der Presets braucht sie nicht): `quality` ist dann leer."""
    samples = [block_of(p, seed0 + i) for i in range(n)]
    if any(k in (C.RULE_MEDIAN, C.RULE_MEAN, C.RULE_AWARE) for k in rules):
        fcs = forecast_for(p, samples)
    else:                                       # nur das Quantil 0,7 gebraucht (Grundgesamtheit der Presets): ein Modell statt fünf
        X = np.vstack([s.X for s in samples])
        pred = F.forecast_curve(p.learner, p.nu, p.n_train, p.train_seed, X)
        off = np.cumsum([0] + [len(s.X) for s in samples])
        fcs = [F.Forecast(None, None, pred[off[i]:off[i + 1]], None, None) for i in range(n)]
    rows = []
    for i, (s, fc) in enumerate(zip(samples, fcs)):
        rows.append(Row(seed0 + i, {k: rule_moves(s, k, fc).moves / p.n_containers for k in rules}))
    quality = {}
    for key in rules if with_quality else ():
        if key == C.RULE_HELL:
            quality[key] = Quality(0.0, 0.0)
        elif key == C.RULE_LOWEST:
            quality[key] = None
        else:
            quality[key] = quality_of(samples, [_clip(forecast_of(key, fc)) for fc in fcs])
    return Sample(tuple(rows), quality, p.n_containers)


def values(rows, key):
    """Umstapelungen je Container der Regel über die Blöcke (Liste)."""
    return [r.m[key] for r in rows]


def mean_of(rows, key):
    return statistics.fmean(values(rows, key))


def diffs(rows, key, ref=C.BASELINE):
    """Gepaarte Differenz je Block: Regel minus Bezug (negativ = weniger Umstapelungen)."""
    return [r.m[key] - r.m[ref] for r in rows]


def paired(rows, key, ref=C.BASELINE):
    """(Mittel, Standardfehler) der gepaarten Differenz."""
    d = diffs(rows, key, ref)
    se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
    return statistics.fmean(d), se


@dataclass(frozen=True)
class Verdict:
    kind: str            # "better" | "worse" | "unclear"
    diff: float          # Regel minus Bezug, Umstapelungen je Container
    se: float            # Standardfehler der gepaarten Differenz
    pct: float           # Anteil am Bezug in Prozent (negativ = weniger), None bei Bezug 0
    n: int


def verdict(rows, key, ref=C.BASELINE):
    """'Klar' heißt: Unterschied größer als VERDICT_Z Standardfehler der gepaarten Differenz. Sonst 'unclear': lieber kein Urteil als eines im Rauschen."""
    diff, se = paired(rows, key, ref)
    base = mean_of(rows, ref)
    kind = "unclear" if abs(diff) <= C.VERDICT_Z * se else ("better" if diff < 0 else "worse")
    return Verdict(kind, diff, se, 100.0 * diff / base if base else None, len(rows))


@dataclass(frozen=True)
class Distribution:
    better: float        # Anteil der Blöcke mit weniger Umstapelungen als der Bezug
    equal: float
    worse: float
    median: float        # Median der gepaarten Differenz (je Container)
    top_decile_share: float   # Anteil der besten 10 % der Blöcke an der gesamten Ersparnis (None ohne Ersparnis)


def distribution(rows, key, ref=C.BASELINE):
    d = diffs(rows, key, ref)
    n = len(d)
    better = sum(1 for x in d if x < -C.STATIC_TOL) / n
    worse = sum(1 for x in d if x > C.STATIC_TOL) / n
    savings = sorted((-x for x in d), reverse=True)
    total = sum(savings)
    k = max(1, -(-n // 10))
    top = sum(savings[:k]) / total if total > C.STATIC_TOL else None
    return Distribution(better, 1.0 - better - worse, worse, statistics.median(d), top)


# ---------------------------------------------------------------------------------------------------
# Gauß-Kurve der Stapelplanung (Rauschen sigma) auf denselben Blöcken
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussCurve:
    sigmas: tuple
    moves: tuple          # Umstapelungen je Container (Mittel über die Blöcke), Bestfit mit Rauschen sigma
    mae: tuple
    pair_err: tuple
    pure: float           # Bezug (Mittel)
    hell: float           # Hellseher (Mittel)
    n_instances: int


def gauss_curve(block_key, n=C.SAMPLE_INSTANCES, sigmas=C.GAUSS_SIGMAS):
    """Bestfit mit wahrer Abfahrt + Gauß-Rauschen sigma * mittlere Verweilzeit gegen den Bezug, auf den Blöcken mit den Seeds 0 .. n-1 (ohne Ankündigung: sie wird hier nicht gebraucht)."""
    S, H, fp, nc = block_key
    samples = [make_block(S, H, fp, nc, None, i) for i in range(n)]
    d0 = float(np.mean([s.mean_dwell_days for s in samples]))
    pure = float(np.mean([pure_moves(s) / nc for s in samples]))
    hell = float(np.mean([K.run(s.inst, F.true_est(s), K.bestfit).moves / nc for s in samples]))
    moves, maes, pairs = [], [], []
    mul, mub = C.GAUSS_NOISE_SEED
    for sg in sigmas:
        ests = [F.noisy_est(s, sg, mul * i + mub) for i, s in enumerate(samples)]
        moves.append(float(np.mean([K.run(s.inst, e, K.bestfit).moves / nc for s, e in zip(samples, ests)])))
        q = quality_est(samples, [np.array([e[c] for c in range(nc)]) for e in ests], d0)
        maes.append(q.mae)
        pairs.append(q.pair_err)
    return GaussCurve(tuple(sigmas), tuple(moves), tuple(maes), tuple(pairs), pure, hell, n)


@dataclass(frozen=True)
class Tipping:
    kind: str            # "crosses" | "always_better" | "always_worse"
    sigma: float         # Rauschen, ab dem Bestfit den Bezug verliert (lineare Interpolation), nur bei "crosses"
    pair_err: float
    mae: float


def tipping_point(curve):
    """Rauschen (und Paarfehler, MAE), ab dem Bestfit mit verrauschter Abfahrt den Bezug verliert: erster Vorzeichenwechsel von (Bestfit - Bezug) auf dem Raster, linear interpoliert."""
    d = [m - curve.pure for m in curve.moves]
    if d[0] >= 0:
        return Tipping("always_worse", 0.0, 0.0, 0.0)
    for i in range(len(d) - 1):
        if d[i] < 0 <= d[i + 1]:
            f = -d[i] / (d[i + 1] - d[i])
            lerp = lambda a, b: a + f * (b - a)                      # noqa: E731
            return Tipping("crosses", lerp(curve.sigmas[i], curve.sigmas[i + 1]), lerp(curve.pair_err[i], curve.pair_err[i + 1]), lerp(curve.mae[i], curve.mae[i + 1]))
    return Tipping("always_better", curve.sigmas[-1], curve.pair_err[-1], curve.mae[-1])


def equivalent_sigma(curve, moves):
    """Das Gauß-Rauschen, das auf denselben Blöcken dieselben Umstapelungen je Container ergibt (auf der Kurve abgelesen): (sigma, 'inside' | 'below' | 'above')."""
    m = curve.moves
    if moves < m[0]:
        return curve.sigmas[0], "below"
    for i in range(len(m) - 1):
        if m[i] <= moves <= m[i + 1]:
            if m[i + 1] == m[i]:
                return curve.sigmas[i], "inside"
            f = (moves - m[i]) / (m[i + 1] - m[i])
            return curve.sigmas[i] + f * (curve.sigmas[i + 1] - curve.sigmas[i]), "inside"
    return curve.sigmas[-1], "above"


# ---------------------------------------------------------------------------------------------------
# Lernkurve: Umstapelungen über der Trainingsmenge
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningCurve:
    sizes: tuple
    series: dict           # Lerner -> Tupel je Trainingsmenge: Differenz zum Bezug (Umstapelungen je Container, Mittel über die Blöcke beim eingestellten Trainings-Seed)
    band: dict             # Lerner -> Tupel je Trainingsmenge: (kleinste, größte) Differenz über die Trainings-Seeds (gleich dem Wert, wo nur ein Seed läuft)
    pair_err: tuple        # Paarfehler des eingestellten Lerners je Trainingsmenge (Trainings-Seed des Reglers)
    moves: tuple           # Umstapelungen je Container des eingestellten Lerners je Trainingsmenge
    n_instances: int
    seeds: tuple


def _curve_eval(samples, learner, nu, size, seed, nc, pair=False):
    X = np.vstack([s.X for s in samples])
    pred = _clip(F.forecast_curve(learner, nu, size, seed, X))
    off = np.cumsum([0] + [len(s.X) for s in samples])
    preds = [pred[off[i]:off[i + 1]] for i in range(len(samples))]
    mv = float(np.mean([K.run(s.inst, F.est_plain(s.t_arr + p), K.bestfit).moves / nc for s, p in zip(samples, preds)]))
    return mv, (quality_of(samples, preds).pair_err if pair else None)


def learning_curve(p, n=C.SAMPLE_INSTANCES):
    """Differenz zum Bezug über der Trainingsmenge für den eingestellten Lerner (Prognose: Quantil 0,7), den anderen Lerner und das Gruppenmittel (Mittelwert je Art, ohne Ankündigung); dazu bei
    kleinen Trainingsmengen das Band über weitere Trainings-Seeds: eine Ziehung der Historie ist ein Glücksspiel."""
    nc = p.n_containers
    samples = [block_of(p, i) for i in range(n)]
    pure = float(np.mean([pure_moves(s) / nc for s in samples]))
    seeds = tuple((p.train_seed + j) % (C.TRAIN_SEED_RANGE[1] + 1) for j in range(C.LEARNING_CURVE_EXTRA_SEEDS + 1))
    series, band, moves, pair_err = {}, {}, (), ()
    for learner in C.LEARNERS + (C.LEARNER_GROUP,):
        ser, bnd = [], []
        for size in C.TRAIN_SIZES:
            use = seeds if size < C.LEARNING_CURVE_BAND_BELOW else seeds[:1]
            vals = [_curve_eval(samples, learner, p.nu, size, sd, nc, pair=(learner == p.learner and sd == p.train_seed)) for sd in use]
            ser.append(vals[0][0] - pure)
            bnd.append((min(v[0] for v in vals) - pure, max(v[0] for v in vals) - pure))
            if learner == p.learner:
                moves += (vals[0][0],)
                pair_err += (vals[0][1],)
        series[learner], band[learner] = tuple(ser), tuple(bnd)
    return LearningCurve(C.TRAIN_SIZES, series, band, pair_err, moves, n, seeds)


# ---------------------------------------------------------------------------------------------------
# Meldung
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Diagnosis:
    kind: str                 # "helps" | "tipped" | "unclear"
    verdict: object           # Verdict der gewählten Regel gegen den Bezug (Stichprobe)
    standard_only: bool       # keine Ankündigung: nur Standardmerkmale
    few_data: bool            # weniger Trainingscontainer als FEW_DATA_BELOW
    block_rule: int           # Umstapelungen der gewählten Regel am gezeigten Block
    block_ref: int            # Umstapelungen des Bezugs am gezeigten Block


def diagnose(rule_key, rows, outcomes, p):
    v = verdict(rows, rule_key)
    kind = {"better": "helps", "worse": "tipped", "unclear": "unclear"}[v.kind]
    return Diagnosis(kind, v, p.nu is None, p.n_train < C.FEW_DATA_BELOW, outcome_of(outcomes, rule_key).moves, outcome_of(outcomes, C.BASELINE).moves)


def diagnosis_text(d, rule_key, seed):
    """Deutscher Satz der bedingten Meldung; die Zusätze für 'nur Standardmerkmale' und 'zu wenig Trainingsdaten' hängen an jeder Art."""
    v = d.verdict
    name = C.RULE_SHORT[rule_key]
    if d.kind == "helps":
        pct = f"{abs(v.pct):.0f} % weniger" if v.pct is not None else f"{abs(v.diff):.3f} weniger"
        head = f"Die Prognose lohnt sich: {name} braucht im Mittel über {v.n} Blöcke {pct} Umstapelungen als der Ausgleich ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f})."
    elif d.kind == "tipped":
        pct = f"{v.pct:.0f} % mehr" if v.pct is not None else f"{v.diff:.3f} mehr"
        head = f"Die Prognose kippt: {name} braucht im Mittel über {v.n} Blöcke {pct} Umstapelungen als der Ausgleich ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}). Der einfache Ausgleich ist besser."
    else:
        head = (f"Kein klarer Unterschied: {name} und der Ausgleich liegen im Mittel über {v.n} Blöcke innerhalb des Rauschens ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}).")
    text = f"{head} Am gezeigten Block (Seed {seed}): {d.block_rule} gegen {d.block_ref} Umstapelungen, einzelne Blöcke schwanken."
    if d.standard_only:
        text += (" Nur Standardmerkmale (Art, Kunde, Zoll, Wochenende, Vorlauf): in diesem Modell bleibt so viel Verweilzeit unerklärt, dass selbst die exakte Kenntnis der bedingten Verteilung kaum hilft "
                 "(gemessen bei 6 × 5, 80 % und 300 Containern: +0,043 mit Median, -0,004 mit Quantil 0,7 gegen den Ausgleich). Erst eine verlässliche Ankündigung trägt.")
    if d.few_data:
        text += (f" Weniger als {C.FEW_DATA_BELOW} Trainingscontainer: das Modell lernt zu wenig, das Ergebnis hängt stark an der Ziehung der Historie (Trainings-Seed ändern).")
    return text


# ---------------------------------------------------------------------------------------------------
# Hilfen für die Blockansicht
# ---------------------------------------------------------------------------------------------------
def state_at(instance, result, event_index):
    """Blockzustand nach `event_index` Ereignissen (0 = leer vor dem ersten Ereignis): (stacks, moved, arrived, step); result muss mit record=True gerechnet sein."""
    if event_index == 0:
        return tuple(() for _ in range(instance.n_stacks)), (), None, None
    step = result.steps[event_index - 1]
    return step.stacks, step.moved, (step.container if step.kind == "A" else None), step


def suggested_event(result):
    """Ereignisnummer (1-basiert), an der die Blockansicht startet: die Abholung mit Umstapelung bei der höchsten Belegung; gibt es keine, der Zeitpunkt mit der höchsten Belegung."""
    best_k, best_occ = None, -1
    for k, step in enumerate(result.steps, start=1):
        if step.kind == "D" and step.moved:
            occ = sum(len(s) for s in step.stacks)
            if occ > best_occ:
                best_k, best_occ = k, occ
    if best_k is not None:
        return best_k
    return max(range(1, len(result.steps) + 1), key=lambda k: (sum(len(s) for s in result.steps[k - 1].stacks), -k))


def describe_step(step, event_index, n_events):
    head = f"Ereignis {event_index} von {n_events}: "
    if step is None:
        return head + "Der Block ist noch leer."
    if step.kind == "A":
        return head + f"Container {step.container} kommt an und wird auf Stapel {step.stack + 1} gelegt."
    if not step.moved:
        return head + f"Container {step.container} wird aus Stapel {step.stack + 1} abgeholt, ohne Umstapelung."
    n = len(step.moved)
    return head + (f"Container {step.container} wird aus Stapel {step.stack + 1} abgeholt, dafür {'muss 1 Container' if n == 1 else f'müssen {n} Container'} umgestapelt werden.")
