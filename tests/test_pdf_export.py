"""PDF-Export: Sonderzeichen mit den GENAUEN Zeichen (fpdf2 stürzt bei Gedankenstrich, Euro und griechischen Buchstaben ab), Sätze, Inhalt Abschnitt für Abschnitt, Randfälle."""

import re

import pytest

import vwz_constants as C
import vwz_evaluation as E
from vwz_pdf_export import generate_vwz_pdf, pdf_text, short_name, verdict_sentence

LOW, MED, Q70 = C.RULE_LOWEST, C.RULE_MEDIAN, C.RULE_Q70
P = E.Params(6, 5, 80, 100, 2, 3, "linear", 5)


def texts(data):
    """Alle Textstücke des (unkomprimierten) PDFs als Liste, Latin-1 gelesen, PDF-Escapes aufgelöst."""
    raw = re.findall(rb"\((.*?)\)\s*Tj", data)
    return [t.decode("latin-1").replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\") for t in raw]


@pytest.fixture(scope="module")
def parts():
    outs = E.run_rules(P, 3)
    sm = E.sample(P, n=8)
    return dict(outs=outs, sm=sm, diag=E.diagnose(Q70, sm.rows, outs, P), gauss=E.gauss_curve(P.block_key, n=6), lc=E.learning_curve(P, n=4))


def build(parts, view=Q70, p=P, **kw):
    args = dict(rows=parts["sm"].rows, sample=parts["sm"], gauss=parts["gauss"], lc=parts["lc"])
    args.update(kw)
    return generate_vwz_pdf(p, 3, view, parts["outs"], parts["diag"], compress=False, **args)


EXPECTED = {"–": "-", "—": "-", "−": "-", "€": "EUR", "Σ": "Summe", "δ": "Delta", "Δ": "Delta", "σ": "sigma", "ν": "nu", "≥": ">=", "≤": "<=", "→": "->", "≈": "ca.", "„": '"', "“": '"', "’": "'",
            "·": "-", "±": "+-", "…": "...", "⚠️": "(!)", "⚠": "(!)", "✅": "", "ℹ️": ""}


@pytest.mark.parametrize("char,replacement", list(EXPECTED.items()))
def test_pdf_text_replaces_every_known_troublemaker_with_a_readable_equivalent(char, replacement):
    out = pdf_text(f"a{char}b")
    out.encode("latin-1")
    assert out == f"a{replacement}b"


def test_pdf_text_keeps_umlauts_and_times_sign_and_replaces_unknown():
    assert pdf_text("Füllgrad äöüß ÄÖÜ × 3") == "Füllgrad äöüß ÄÖÜ × 3" and pdf_text("日本語").encode("latin-1") == b"???" and "?" in pdf_text("🔮 Verweilzeit")


def test_short_names_have_no_emoji_and_survive_latin_1():
    assert [short_name(k) for k in C.RULE_KEYS] == ["Niedrigster Stapel", "Bestfit Median", "Bestfit Quantil 0,7", "Unsicherheitsbewusst", "Bestfit Mittelwert", "Niedr. Stapel + Prognose", "Hellseher"]
    assert all(pdf_text(short_name(k)) == short_name(k) for k in C.RULE_KEYS)


def rows_from_diffs(diffs, rein=1.0):
    return tuple(E.Row(i, {k: (rein if k == LOW else rein + d) for k in C.RULE_KEYS}) for i, d in enumerate(diffs))


def test_verdict_sentence_covers_all_states_and_survives_latin_1():
    better = rows_from_diffs([-0.5 + 0.3 * (i % 2 * 2 - 1) for i in range(10)])
    worse = rows_from_diffs([0.5 + 0.3 * (i % 2 * 2 - 1) for i in range(10)])
    unclear = rows_from_diffs([0.05 + 0.3 * (i % 2 * 2 - 1) for i in range(10)])
    sb = verdict_sentence(better, Q70)
    assert sb == "Bestfit Quantil 0,7 gegen Ausgleich: im Mittel 50 % weniger Umstapelungen (-0.500 je Container, Standardfehler 0.100); an 0 % der Blöcke ist es umgekehrt."
    sw = verdict_sentence(worse, MED)
    assert sw == "Bestfit Median gegen Ausgleich: im Mittel 50 % mehr Umstapelungen (+0.500 je Container, Standardfehler 0.100); an 0 % der Blöcke ist es besser."
    su = verdict_sentence(unclear, Q70)
    assert su.startswith("Bestfit Quantil 0,7 gegen Ausgleich: kein klarer Unterschied, die Differenz (+0.050 je Container, gemittelt über die Blöcke) liegt innerhalb des Rauschens (Standardfehler 0.100);") \
        and "besser an 50 %, schlechter an 50 % der Blöcke." in su
    for s in (sb, sw, su):
        s.encode("latin-1")
    assert "im Mittel 0.100 weniger Umstapelungen" in verdict_sentence(rows_from_diffs([-0.1, -0.1, -0.11, -0.09, -0.1], rein=0.0), Q70)          # Bezug 0: kein Prozentwert, Betrag stattdessen
    assert "im Mittel 0.100 mehr Umstapelungen" in verdict_sentence(rows_from_diffs([0.1, 0.1, 0.11, 0.09, 0.1], rein=0.0), Q70)


def test_pdf_is_a_valid_document_with_all_sections(parts):
    data = build(parts)
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")
    t = texts(data)
    joined = "\n".join(t)
    for needle in ("Verweilzeit lernen statt annehmen", "Szenario", "Zusammenfassung", "Regelvergleich (dieser Block)", "Stichprobe und Urteil", "Kipppunkt: wie gut muss die Prognose sein?",
                   "Lernkurve: wie viele Trainingsdaten?", "Hinweise zum Modell"):
        assert needle in t, needle
    assert "6 Stapel, Höhe 5, Füllgrad 80 %, 100 Container" in joined and "mittel (nu 0,6): Fehler der Ankündigung typisch 60 % (log-Einheiten)" in joined
    assert "Lineare Regression, 1000 Trainingscontainer, Trainings-Seed 5" in joined and "Bestfit Quantil 0,7" in joined and "Block-Seed" in joined
    assert E.diagnosis_text(parts["diag"], Q70, 3)[:60] in " ".join(t)                                                    # die Meldung der App steht in der Zusammenfassung


def test_pdf_tables_carry_the_numbers_of_the_run(parts):
    t = texts(build(parts))
    outs = {o.key: o for o in parts["outs"]}
    n = P.n_containers
    assert f"{outs[Q70].moves} Umstapelungen ({outs[Q70].moves / n:.3f} je Container)" in t and f"{outs[LOW].moves} Umstapelungen ({outs[LOW].moves / n:.3f} je Container)" in t
    row = [str(outs[Q70].moves), f"{outs[Q70].moves / n:.3f}", f"{(outs[Q70].moves - outs[LOW].moves) / n:+.3f}", f"{outs[Q70].quality.mae:.2f}", f"{outs[Q70].quality.pair_err * 100:.0f}"]
    start = t.index("Regelvergleich (dieser Block)")
    i = t.index("Bestfit Quantil 0,7", start)
    assert t[i:i + 6] == ["Bestfit Quantil 0,7"] + row
    j = t.index("Niedrigster Stapel", start)
    assert t[j:j + 6] == ["Niedrigster Stapel", str(outs[LOW].moves), f"{outs[LOW].moves / n:.3f}", "-", "-", "-"]                 # Bezug: Delta und Güte leer
    k = t.index("Hellseher", t.index("Regelvergleich (dieser Block)") + 1)
    assert t[k + 4:k + 6] == ["0.00", "0"]
    g = parts["gauss"]
    assert f"{g.moves[3]:.3f}" in t and f"{g.pure:.3f}" in " ".join(t) and "Mittel über 6 Blöcke (Seeds 0-5)" in " ".join(t)
    lc = parts["lc"]
    assert f"{lc.series['gruppe'][0]:+.3f}" in t and "Trainings-Seed 5." in " ".join(t)
    sm = parts["sm"]
    assert f"{E.mean_of(sm.rows, Q70):.3f}" in t and f"{E.paired(sm.rows, Q70)[1]:.3f}" in t and "Basis: 8 Blöcke (Seeds 0-7, nicht der eingestellte Seed)" in " ".join(t)


def test_pdf_page_breaks_only_when_a_section_does_not_fit(parts):
    """Ohne Stichprobe und Kurven passt alles auf eine Seite, mit allen Abschnitten sind es drei (Abschnitte werden nicht zerschnitten, aber auch nicht ohne Not auf eine neue Seite geschoben)."""
    pages = lambda d: len(re.findall(rb"/Type /Page\b", d))                                    # noqa: E731
    assert pages(build(parts, rows=None, sample=None, gauss=None, lc=None)) == 1 and pages(build(parts)) == 3


def test_pdf_view_rule_and_optional_sections(parts):
    full = "\n".join(texts(build(parts, view=MED)))
    assert "Gewählte Regel" in full and "Bestfit Median" in full
    bare = texts(build(parts, rows=None, sample=None, gauss=None, lc=None))
    assert "Stichprobe und Urteil" not in bare and "Kipppunkt: wie gut muss die Prognose sein?" not in bare and "Lernkurve: wie viele Trainingsdaten?" not in bare and "Hinweise zum Modell" in bare
    no_sample = texts(build(parts, sample=None))
    assert "Stichprobe und Urteil" in no_sample and "-" in no_sample


def test_pdf_kipppunkt_sentence_for_all_kinds(parts):
    crosses = " ".join(texts(build(parts)))                                                   # Sätze in Anmerkungen sind über mehrere Zeilen umbrochen
    assert "schlägt den Ausgleich bis sigma =" in crosses and "falsch geordnete Paare (MAE" in crosses
    g = parts["gauss"]
    better = E.GaussCurve(g.sigmas, tuple(x * 0 + 0.1 for x in g.moves), g.mae, g.pair_err, g.pure, g.hell, g.n_instances)
    worse = E.GaussCurve(g.sigmas, tuple(x * 0 + 5.0 for x in g.moves), g.mae, g.pair_err, g.pure, g.hell, g.n_instances)
    assert "Bestfit gewinnt überall." in "\n".join(texts(build(parts, gauss=better))) and "Bestfit verliert überall." in "\n".join(texts(build(parts, gauss=worse)))


def test_pdf_compression_switch_and_special_characters_in_settings(parts):
    small = generate_vwz_pdf(P, 3, Q70, parts["outs"], parts["diag"], rows=parts["sm"].rows, sample=parts["sm"], gauss=parts["gauss"], lc=parts["lc"])
    big = build(parts)
    assert len(small) < len(big) and small.startswith(b"%PDF")
    none = E.Params(6, 5, 80, 100, 0, 0, "boosting", 5)                                        # 'keine' Ankündigung, kleinste Trainingsmenge, anderer Lerner
    t = "\n".join(texts(generate_vwz_pdf(none, 3, Q70, parts["outs"], parts["diag"], compress=False)))
    assert "Ankündigung" in t and "keine" in t and "Gradient Boosting, 30 Trainingscontainer" in t and "Fehler der Ankündigung" not in t
