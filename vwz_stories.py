"""Abnahmekriterien der Presets: Welche Geschichte erzählt jedes Beispielszenario, und woran erkennt man, dass sie trägt?

Einzige Quelle für `tools/tune_presets.py` (Abstimmung) und `tests/test_preset_stories.py` (Abnahme). Zwei Ebenen, weil das Ergebnis am EINZELNEN Block mit der Trainingshistorie schwankt (Messreihe,
Korrektur 6): `criteria` prüft die Geschichte im MITTEL über viele Blöcke (Grundgesamtheit, das Hauptkriterium), `holds` am einen Block, den das Preset zeigt, mit ganzzahligen Schwellen und Abstand.
Alle Kriterien tragen einen Wert der Kategorie 'Umstapelungen je Container' (Blockmittel) beziehungsweise ganze Umstapelungen (Block); Regeln der Grundgesamtheit: Bezug, gelernt (Bestfit, Quantil 0,7), Hellseher.
Deterministisch, ohne Löser und Zeitgrenze: die Kriterien sind CI-robust (das Boosting ist bei gleichen Eingaben deterministisch; die Schwellen liegen weit von den Messwerten)."""

import vwz_constants as C
import vwz_evaluation as E

REIN, GEL, HELL = C.RULE_LOWEST, C.RULE_Q70, C.RULE_HELL
POPULATION_RULES = (REIN, GEL, HELL)

# Kennzahlen, an denen 'typisch' gemessen wird: (Preset, Regel)
TYPICAL = tuple((name, key) for name in C.PRESETS for key in POPULATION_RULES)


def criteria(name, rows):
    """Kriterien über die Blöcke der Grundgesamtheit (Mittel je Container). Rückgabe: Liste (erfüllt, Text)."""
    rein, hell = E.mean_of(rows, REIN), E.mean_of(rows, HELL)
    v = E.verdict(rows, GEL)
    d = v.diff
    if name == "Nur Standard":
        return [(d >= 0.03, f"gelernt minus Ausgleich >= +0,03: {d:+.3f}"), (v.kind == "worse", f"Urteil 'kippt' (mehr als zwei Standardfehler, SE {v.se:.3f}): {v.kind}")]
    if name == "Mit Ankündigung":
        return [(d <= -0.06, f"gelernt minus Ausgleich <= -0,06: {d:+.3f}"), (v.kind == "better", f"Urteil 'Vorteil' (SE {v.se:.3f}): {v.kind}"),
                (hell <= 0.4 * rein, f"Hellseher <= 40 % des Ausgleichs: {hell:.3f} gegen {rein:.3f}")]
    if name == "Zu wenig Daten":
        return [(d >= 0.06, f"gelernt minus Ausgleich >= +0,06: {d:+.3f}"), (v.kind == "worse", f"Urteil 'kippt' (SE {v.se:.3f}): {v.kind}")]
    if name == "Sehr verlässlich":
        return [(d <= -0.18, f"gelernt minus Ausgleich <= -0,18: {d:+.3f}"), (v.kind == "better", f"Urteil 'Vorteil' (SE {v.se:.3f}): {v.kind}")]
    if name == "Voller Block":
        return [(d <= -0.08, f"gelernt minus Ausgleich <= -0,08: {d:+.3f}"), (v.kind == "better", f"Urteil 'Vorteil' (SE {v.se:.3f}): {v.kind}"),
                (rein >= 1.0, f"Ausgleich >= 1,0 je Container (voller Block ist teurer): {rein:.3f}")]
    raise KeyError(name)


def block_criteria(name, moves):
    """Kriterien am gezeigten Block: `moves` = ganze Umstapelungen {RULE_LOWEST, RULE_Q70, RULE_HELL}. Rückgabe: Liste (erfüllt, Text)."""
    rein, gel, hell = moves[REIN], moves[GEL], moves[HELL]
    if name == "Nur Standard":
        return [(gel >= rein + 5, f"gelernt >= Ausgleich + 5: {gel} gegen {rein}")]
    if name == "Mit Ankündigung":
        return [(gel <= rein - 8, f"gelernt <= Ausgleich - 8: {gel} gegen {rein}"), (hell <= 0.4 * rein, f"Hellseher <= 40 % des Ausgleichs: {hell} gegen {rein}")]
    if name == "Zu wenig Daten":
        return [(gel >= rein + 20, f"gelernt >= Ausgleich + 20: {gel} gegen {rein}")]
    if name == "Sehr verlässlich":
        return [(gel <= 0.8 * rein, f"gelernt <= 80 % des Ausgleichs: {gel} gegen {rein}")]
    if name == "Voller Block":
        return [(rein >= 300, f"Ausgleich >= 300: {rein}"), (gel <= rein - 25, f"gelernt <= Ausgleich - 25: {gel} gegen {rein}")]
    raise KeyError(name)


def holds(name, moves):
    """Gilt die Geschichte an dem EINEN Block, den das Preset zeigt?"""
    return all(ok for ok, _ in block_criteria(name, moves))


def key_values(name, rows):
    """Die Kennzahlen dieses Presets aus TYPICAL als {Regel: Mittel je Container}."""
    return {k: E.mean_of(rows, k) for n, k in TYPICAL if n == name}
