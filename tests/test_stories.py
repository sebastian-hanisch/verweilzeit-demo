"""Die Preset-Kriterien einzeln an ihren Schwellen: künstliche Werte, bei denen genau ein Kriterium kippt. Die Schwellen stehen fest im Test (sie werden nicht aus dem Code gelesen). Die Schwellen der
Mittelwerte werden mit 0,001 Abstand geprüft (Gleitkommagrenze), die der ganzen Umstapelungen exakt."""

import pytest

import vwz_constants as C
import vwz_evaluation as E
import vwz_stories as ST

REIN, GEL, HELL = C.RULE_LOWEST, C.RULE_Q70, C.RULE_HELL


def rows(rein, diff, hell=0.2, n=10, spread=0.3):
    """n Blöcke: Bezug `rein`, gelernt mit gepaarter Differenz diff +- spread (je zur Hälfte, Standardfehler spread / 3), Hellseher `hell`."""
    out = []
    for i in range(n):
        d = diff + (spread if i % 2 else -spread)
        out.append(E.Row(i, {REIN: rein, GEL: rein + d, HELL: hell}))
    return tuple(out)


def flags(name, r):
    return [ok for ok, _ in ST.criteria(name, r)]


def bflags(name, rein, gel, hell=0):
    return [ok for ok, _ in ST.block_criteria(name, {REIN: rein, GEL: gel, HELL: hell})]


def test_unknown_preset_raises():
    for f in (lambda: ST.criteria("unbekannt", rows(1, 0)), lambda: ST.block_criteria("unbekannt", {REIN: 1, GEL: 1, HELL: 1}), lambda: ST.holds("unbekannt", {REIN: 1, GEL: 1, HELL: 1})):
        with pytest.raises(KeyError):
            f()


def test_nur_standard_thresholds():
    assert flags("Nur Standard", rows(0.86, 0.5)) == [True, True]
    assert flags("Nur Standard", rows(0.86, 0.031))[0] is True and flags("Nur Standard", rows(0.86, 0.029))[0] is False
    assert flags("Nur Standard", rows(0.86, 0.5, spread=1.0))[1] is False                        # Differenz 0,5 bei Standardfehler 0,33: kein klares Urteil
    assert flags("Nur Standard", rows(0.86, -0.5))[1] is False                                   # klar besser ist nicht 'kippt'
    assert bflags("Nur Standard", 238, 243) == [True] and bflags("Nur Standard", 238, 242) == [False]


def test_mit_ankuendigung_thresholds():
    assert flags("Mit Ankündigung", rows(0.86, -0.3, hell=0.25)) == [True, True, True]
    assert flags("Mit Ankündigung", rows(0.86, -0.061))[0] is True and flags("Mit Ankündigung", rows(0.86, -0.059))[0] is False
    assert flags("Mit Ankündigung", rows(0.86, -0.3, spread=1.5))[1] is False and flags("Mit Ankündigung", rows(0.86, 0.3))[1] is False
    assert flags("Mit Ankündigung", rows(1.0, -0.3, hell=0.4))[2] is True and flags("Mit Ankündigung", rows(1.0, -0.3, hell=0.41))[2] is False           # 40 % des Ausgleichs
    assert bflags("Mit Ankündigung", 100, 92, 40) == [True, True] and bflags("Mit Ankündigung", 100, 93, 40)[0] is False and bflags("Mit Ankündigung", 100, 92, 41)[1] is False


def test_zu_wenig_daten_thresholds():
    assert flags("Zu wenig Daten", rows(0.86, 0.5)) == [True, True]
    assert flags("Zu wenig Daten", rows(0.86, 0.061))[0] is True and flags("Zu wenig Daten", rows(0.86, 0.059))[0] is False
    assert flags("Zu wenig Daten", rows(0.86, 0.5, spread=1.0))[1] is False
    assert bflags("Zu wenig Daten", 238, 258) == [True] and bflags("Zu wenig Daten", 238, 257) == [False]


def test_sehr_verlaesslich_thresholds():
    assert flags("Sehr verlässlich", rows(0.86, -0.5)) == [True, True]
    assert flags("Sehr verlässlich", rows(0.86, -0.181))[0] is True and flags("Sehr verlässlich", rows(0.86, -0.179))[0] is False
    assert flags("Sehr verlässlich", rows(0.86, -0.5, spread=1.5))[1] is False
    assert bflags("Sehr verlässlich", 100, 80) == [True] and bflags("Sehr verlässlich", 100, 81) == [False]                          # 80 % des Ausgleichs


def test_voller_block_thresholds():
    assert flags("Voller Block", rows(1.1, -0.5)) == [True, True, True]
    assert flags("Voller Block", rows(1.1, -0.081))[0] is True and flags("Voller Block", rows(1.1, -0.079))[0] is False
    assert flags("Voller Block", rows(1.1, -0.5, spread=1.5))[1] is False
    assert flags("Voller Block", rows(1.0, -0.5))[2] is True and flags("Voller Block", rows(0.99, -0.5))[2] is False
    assert bflags("Voller Block", 300, 275) == [True, True] and bflags("Voller Block", 300, 276)[1] is False and bflags("Voller Block", 299, 250)[0] is False


def test_holds_applies_all_block_criteria():
    assert ST.holds("Mit Ankündigung", {REIN: 238, GEL: 222, HELL: 79}) and not ST.holds("Mit Ankündigung", {REIN: 238, GEL: 231, HELL: 79})
    assert not ST.holds("Mit Ankündigung", {REIN: 238, GEL: 222, HELL: 100})                          # ein Kriterium verletzt: nicht erfüllt (nicht 'irgendeins')
    assert ST.holds("Voller Block", {REIN: 355, GEL: 310, HELL: 148}) and not ST.holds("Voller Block", {REIN: 355, GEL: 335, HELL: 148})


def test_typical_and_key_values_cover_every_preset():
    assert ST.TYPICAL == tuple((n, k) for n in C.PRESETS for k in (REIN, GEL, HELL)) and ST.POPULATION_RULES == (REIN, GEL, HELL)
    kv = ST.key_values("Sehr verlässlich", rows(0.86, -0.5, hell=0.25))
    assert kv == {REIN: pytest.approx(0.86), GEL: pytest.approx(0.36), HELL: pytest.approx(0.25)}
    assert set(ST.key_values("Nur Standard", rows(1, 0))) == {REIN, GEL, HELL}
