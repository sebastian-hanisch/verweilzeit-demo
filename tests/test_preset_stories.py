"""Abnahme der Presets an ECHTEN Daten: jede Geschichte trägt im Mittel über 200 Blöcke (Seeds 100-299) UND an dem einen Block, den das Preset zeigt; der gezeigte Block ist typisch, nicht der schönste
Einzelfall; über 20 Trainingshistorien behält die Grundgesamtheit ihr Vorzeichen (langsam). Deterministisch (kein Löser, keine Zeitgrenze)."""

import pytest

import vwz_constants as C
import vwz_evaluation as E
import vwz_stories as ST

POP_SEED0, POP_N = 100, 200
NAMES = list(C.PRESETS)
SHOWN = 268
REIN, GEL, HELL = C.RULE_LOWEST, C.RULE_Q70, C.RULE_HELL
_POP = {}


def params(name, train_seed=None):
    p = E.params_from_preset(C.PRESETS[name])
    return p if train_seed is None else E.Params(*p.key[:-1], train_seed)


def population(name, train_seed=None):
    key = (name, train_seed)
    if key not in _POP:
        _POP[key] = E.sample(params(name, train_seed), n=POP_N, seed0=POP_SEED0, rules=ST.POPULATION_RULES, with_quality=False)
    return _POP[key]


def block_moves(name):
    outs = E.run_rules(params(name), C.PRESETS[name]["seed"], record=False)
    return {k: E.outcome_of(outs, k).moves for k in ST.POPULATION_RULES}


@pytest.mark.parametrize("name", NAMES)
def test_story_holds_on_average_over_the_population(name):
    for ok, text in ST.criteria(name, population(name).rows):
        assert ok, f"{name}: {text}"


@pytest.mark.parametrize("name", NAMES)
def test_story_holds_at_the_block_the_preset_shows(name):
    for ok, text in ST.block_criteria(name, block_moves(name)):
        assert ok, f"{name}: {text}"
    assert ST.holds(name, block_moves(name))


def test_the_preset_seed_lies_outside_the_kernabschnitt_sample_and_inside_the_population():
    assert all(p["seed"] == SHOWN and C.SAMPLE_INSTANCES <= p["seed"] and POP_SEED0 <= p["seed"] < POP_SEED0 + POP_N for p in C.PRESETS.values())


@pytest.mark.parametrize("name", NAMES)
def test_the_shown_block_is_typical_for_every_key_measure(name):
    """Jede Kennzahl des gezeigten Blocks liegt zwischen dem 10. und 90. Perzentil der Grundgesamtheit (je Container)."""
    rows = population(name).rows
    shown = rows[SHOWN - POP_SEED0]
    for n, key in ST.TYPICAL:
        if n != name:
            continue
        vals = sorted(E.values(rows, key))
        lo, hi = vals[int(0.1 * len(vals))], vals[int(0.9 * len(vals)) - 1]
        assert lo <= shown.m[key] <= hi, (name, key, shown.m[key], (lo, hi))


def test_the_stories_differ_between_presets():
    assert not all(ok for ok, _ in ST.criteria("Mit Ankündigung", population("Nur Standard").rows))
    assert not all(ok for ok, _ in ST.criteria("Nur Standard", population("Mit Ankündigung").rows))
    assert not all(ok for ok, _ in ST.criteria("Zu wenig Daten", population("Mit Ankündigung").rows))
    assert not all(ok for ok, _ in ST.criteria("Sehr verlässlich", population("Mit Ankündigung").rows))
    assert not all(ok for ok, _ in ST.criteria("Voller Block", population("Nur Standard").rows))
    assert not ST.holds("Nur Standard", block_moves("Mit Ankündigung")) and not ST.holds("Zu wenig Daten", block_moves("Sehr verlässlich"))


def test_the_population_reproduces_the_messreihe():
    """presets_final300.json (hafen-planung/messreihe_verweilzeit): Differenz gelernt minus Ausgleich der Grundgesamtheit, Standardfehler, Ausgleich, Hellseher; Toleranz 0,003 je Container."""
    expected = {"Nur Standard": (+0.0659, 0.0075, 0.8598, 0.2520), "Mit Ankündigung": (-0.1146, 0.0061, 0.8598, 0.2520), "Zu wenig Daten": (+0.1164, 0.0075, 0.8598, 0.2520),
                "Sehr verlässlich": (-0.2619, 0.0054, 0.8598, 0.2520), "Voller Block": (-0.1431, 0.0065, 1.1361, 0.4963)}
    for name, (diff, se, rein, hell) in expected.items():
        rows = population(name).rows
        d, s = E.paired(rows, GEL)
        assert d == pytest.approx(diff, abs=0.003) and s == pytest.approx(se, abs=0.002), name
        assert E.mean_of(rows, REIN) == pytest.approx(rein, abs=0.003) and E.mean_of(rows, HELL) == pytest.approx(hell, abs=0.003), name


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_population_keeps_its_sign_over_20_training_histories(name):
    """Trainings-Seeds 5 bis 24: das Vorzeichen der Grundgesamtheit hält in ALLEN 20 Ziehungen (Messreihe); die Schwellen der Kriterien halten nur teilweise, am schlechtesten bei 'Zu wenig Daten'
    (30 Trainingscontainer sind ein Glücksspiel): dort mindestens die Hälfte. Am gezeigten Block hält die Geschichte in der Mehrzahl."""
    diffs, holds_pop, holds_block = [], 0, 0
    for h in range(5, 25):
        rows = population(name, h).rows
        diffs.append(E.paired(rows, GEL)[0])
        holds_pop += all(ok for ok, _ in ST.criteria(name, rows))
        r = rows[SHOWN - POP_SEED0]
        holds_block += ST.holds(name, {k: round(r.m[k] * C.PRESETS[name]["n_containers"]) for k in ST.POPULATION_RULES})
    if name in ("Nur Standard", "Zu wenig Daten"):
        assert all(d > 0.01 for d in diffs), (name, diffs)                        # kippt in jeder Ziehung
    else:
        assert all(d < -0.05 for d in diffs), (name, diffs)                       # Vorteil in jeder Ziehung
    assert holds_pop >= (10 if name == "Zu wenig Daten" else 18) and holds_block >= (12 if name == "Zu wenig Daten" else 16), (name, holds_pop, holds_block)
