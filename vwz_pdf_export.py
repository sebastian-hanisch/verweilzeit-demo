"""PDF-Export des Ergebnisses (fpdf2, Helvetica-Kernschrift, nur Text und Tabellen).

Die Kernschriften kennen nur Latin-1: Umlaute und "×" sind erlaubt, aber "–" (Gedankenstrich), "−" (Minuszeichen), "€", "σ", "ν", "Σ", "≥", "≤", Emoji usw. lassen fpdf2 abstürzen. Deshalb läuft jeder Text
durch pdf_text(); Regeln erscheinen mit ihren Kurznamen ohne Emoji."""

import time

import vwz_constants as C
import vwz_evaluation as E

_REPLACEMENTS = {
    "–": "-", "—": "-", "‑": "-", "−": "-", "Σ": "Summe", "δ": "Delta", "Δ": "Delta", "σ": "sigma", "ν": "nu", "≥": ">=", "≤": "<=", "→": "->", "≈": "ca.", "€": "EUR", "·": "-",
    "“": '"', "”": '"', "„": '"', "’": "'", "‘": "'", "±": "+-", "…": "...", "⚠️": "(!)", "⚠": "(!)", "✅": "", "ℹ️": "",
}


def pdf_text(text):
    """Text für die Helvetica-Kernschrift: bekannte Sonderzeichen ersetzen, den Rest Latin-1-sicher machen."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def short_name(key):
    return C.RULE_SHORT[key]


def verdict_sentence(rows, key):
    """Ein Satz zum Urteil über die Stichprobe (ohne Emoji), wie im Kernabschnitt der App."""
    v = E.verdict(rows, key)
    d = E.distribution(rows, key)
    name = C.RULE_SHORT[key]
    if v.kind == "better":
        amount = f"{abs(v.pct):.0f} % weniger" if v.pct is not None else f"{abs(v.diff):.3f} weniger"
        return f"{name} gegen Ausgleich: im Mittel {amount} Umstapelungen ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}); an {d.worse * 100:.0f} % der Blöcke ist es umgekehrt."
    if v.kind == "worse":
        amount = f"{v.pct:.0f} % mehr" if v.pct is not None else f"{v.diff:.3f} mehr"
        return f"{name} gegen Ausgleich: im Mittel {amount} Umstapelungen ({v.diff:+.3f} je Container, Standardfehler {v.se:.3f}); an {d.better * 100:.0f} % der Blöcke ist es besser."
    return (f"{name} gegen Ausgleich: kein klarer Unterschied, die Differenz ({v.diff:+.3f} je Container, gemittelt über die Blöcke) liegt innerhalb des Rauschens (Standardfehler {v.se:.3f}); "
            f"besser an {d.better * 100:.0f} %, schlechter an {d.worse * 100:.0f} % der Blöcke.")


def generate_vwz_pdf(p, seed, view_key, outcomes, diag, rows=None, sample=None, gauss=None, lc=None, compress=True):
    """Ergebnis der aktuellen Einstellung als PDF: Szenario, Zusammenfassung, Regelvergleich, optional Stichprobe/Urteil, Kipppunkt und Lernkurve, Hinweise.

    `p`: E.Params; `outcomes`: die Outcomes eines Blocks; `diag`: E.Diagnosis; `rows`: Blöcke der Stichprobe (E.Sample.rows) oder None; `sample`: E.Sample (für MAE und Paarfehler); `gauss`:
    E.GaussCurve oder None; `lc`: E.LearningCurve oder None."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    n = p.n_containers
    base = E.outcome_of(outcomes, C.BASELINE)

    pdf = FPDF()
    pdf.set_compression(compress)
    pdf.add_page()

    def line(text, height=7, width=0):
        pdf.cell(width, height, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(text):
        pdf.set_font("Helvetica", "B", 12)
        line(text, 8)
        pdf.set_font("Helvetica", "", 10)

    def pairs(rows_):
        for label, value in rows_:
            pdf.cell(60, 6, pdf_text(label), border=0)
            line(value, 6)

    def table(headers, widths, body):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 230)
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, pdf_text(header), border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 9)
        for row in body:
            for value, width in zip(row, widths):
                pdf.cell(width, 7, pdf_text(str(value)), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(7)

    def keep_together(height):
        """Beginnt einen Abschnitt auf einer neuen Seite, wenn er sonst über den Seitenumbruch liefe (keine halb abgeschnittenen Tabellen)."""
        if pdf.get_y() + height > pdf.h - pdf.b_margin:
            pdf.add_page()

    def note(text, size=8):
        pdf.set_font("Helvetica", "I", size)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 5, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "B", 16)
    line("Verweilzeit lernen statt annehmen", 10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    line(f"Erstellt: {time.strftime('%d.%m.%Y %H:%M')}  -  sebastianhanisch.net", 6)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    heading("Szenario")
    pairs([("Block", f"{p.n_stacks} Stapel, Höhe {p.max_height}, Füllgrad {p.fill_pct} %, {n} Container"),
           ("Ankündigung", C.NU_LABELS[p.announce] if p.nu is None else f"{C.NU_LABELS[p.announce]}: Fehler der Ankündigung typisch {int(round(p.nu * 100))} % (log-Einheiten)"),
           ("Lerner", f"{C.LEARNER_LABELS[p.learner]}, {p.n_train} Trainingscontainer, Trainings-Seed {p.train_seed}"),
           ("Gewählte Regel", short_name(view_key)), ("Block-Seed", str(seed))])
    pdf.ln(3)

    heading("Zusammenfassung")
    note(E.diagnosis_text(diag, view_key, seed), 9)
    pairs([(short_name(o.key), f"{o.moves} Umstapelungen ({o.moves / n:.3f} je Container)") for o in outcomes if o.key in (C.RULE_LOWEST, view_key, C.RULE_HELL)])
    pdf.ln(3)

    heading("Regelvergleich (dieser Block)")
    body = []
    for o in outcomes:
        q = o.quality
        body.append([short_name(o.key), o.moves, f"{o.moves / n:.3f}", "-" if o.key == base.key else f"{(o.moves - base.moves) / n:+.3f}",
                     "-" if q is None else f"{q.mae:.2f}", "-" if q is None else f"{q.pair_err * 100:.0f}"])
    table(["Regel", "Umstapelungen", "je Container", "gegen Ausgleich", "MAE (Standz.)", "falsche Paare %"], [50, 28, 26, 30, 28, 28], body)
    note("Bezug ist der Niedrigster Stapel ohne Prognose (Einlagern und Umstapeln nach niedrigstem Stapel). MAE: mittlerer Betragsfehler der Verweilzeit in mittleren Standzeiten; falsche Paare: Anteil der "
         "gleichzeitig anwesenden Paare, deren Abfahrtsreihenfolge falsch geschätzt ist.")
    pdf.ln(3)

    if rows is not None:
        keep_together(95)
        heading("Stichprobe und Urteil")
        table(["Regel", "je Container", "gegen Ausgleich", "Standardfehler", "MAE", "falsche Paare %"], [50, 26, 30, 30, 26, 28],
              [[short_name(k), f"{E.mean_of(rows, k):.3f}", "-" if k == C.BASELINE else f"{E.paired(rows, k)[0]:+.3f}", "-" if k == C.BASELINE else f"{E.paired(rows, k)[1]:.3f}",
                "-" if sample is None or sample.quality.get(k) is None else f"{sample.quality[k].mae:.2f}",
                "-" if sample is None or sample.quality.get(k) is None else f"{sample.quality[k].pair_err * 100:.0f}"] for k in C.RULE_KEYS])
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 9)
        for key in (C.RULE_Q70, C.RULE_MEDIAN):
            pdf.multi_cell(0, 5, pdf_text("- " + verdict_sentence(rows, key)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        note(f"Basis: {len(rows)} Blöcke (Seeds 0-{len(rows) - 1}, nicht der eingestellte Seed) mit den eingestellten Werten. Klar heißt: Unterschied größer als zwei Standardfehler der gepaarten Differenz.")
        pdf.ln(3)

    if gauss is not None:
        keep_together(85)
        heading("Kipppunkt: wie gut muss die Prognose sein?")
        t = E.tipping_point(gauss)
        note(("Bestfit mit Gauß-Rauschen (Regler der Stapelplanung) schlägt den Ausgleich bis sigma = %.2f Standzeiten, das sind %.0f %% falsch geordnete Paare (MAE %.2f)." % (t.sigma, t.pair_err * 100, t.mae))
             if t.kind == "crosses" else "Kein Kipppunkt im Bereich sigma 0,1 bis 2,0: " + ("Bestfit gewinnt überall." if t.kind == "always_better" else "Bestfit verliert überall."), 9)
        table(["Rauschen sigma", "je Container", "gegen Ausgleich", "MAE", "falsche Paare %"], [36, 32, 36, 30, 36],
              [[f"{s:.2f}", f"{gauss.moves[i]:.3f}", f"{gauss.moves[i] - gauss.pure:+.3f}", f"{gauss.mae[i]:.2f}", f"{gauss.pair_err[i] * 100:.0f}"] for i, s in enumerate(gauss.sigmas)])
        note(f"Mittel über {gauss.n_instances} Blöcke (Seeds 0-{gauss.n_instances - 1}); Ausgleich {gauss.pure:.3f}, Hellseher {gauss.hell:.3f} Umstapelungen je Container.")
        pdf.ln(3)

    if lc is not None:
        keep_together(80)
        heading("Lernkurve: wie viele Trainingsdaten?")
        table(["Trainingscontainer"] + [f"{s}" for s in lc.sizes], [42] + [24] * len(lc.sizes),
              [[C.LEARNER_LABELS[k][:22]] + [f"{v:+.3f}" for v in lc.series[k]] for k in C.LEARNERS + (C.LEARNER_GROUP,)])
        note(f"Differenz zum Ausgleich je Container (negativ = besser), Mittel über {lc.n_instances} Blöcke; Prognose: Quantil 0,7, beim Gruppenmittel der Mittelwert je Art. Trainings-Seed {lc.seeds[0]}.")
        pdf.ln(3)

    keep_together(70)
    heading("Hinweise zum Modell")
    pdf.set_font("Helvetica", "", 9)
    for text in [
        "Alle Merkmale, die Verteilung der Verweilzeit (lognormal, heteroskedastisch) und die Ankündigung mit lognormalem Fehler sind erfunden (synthetisch, keine Echtdaten); die Größenordnung des Nutzens hängt an der angenommenen Streuung.",
        "Die Abholreihenfolge folgt allein der Verweilzeit (keine Bündelung nach Schiff); Training und Test stammen aus derselben Verteilung (kein Saisonwechsel); ein Block, keine Kranwege.",
        "Das Boosting ist selbst gebaut (nur numpy) und nicht abgestimmt; es stimmt in der Messreihe mit scikit-learn auf 0,2 bis 1,1 % (Pinball-Verlust) überein.",
        "Nutzen (Umstapelungen) und Prognosegüte (MAE, falsche Paare) werden bewusst getrennt gezeigt; was eine bessere Prognose kostet (Daten, Erhebung, Pflege), ist Sache des Betreibers.",
    ]:
        pdf.multi_cell(0, 5, pdf_text("- " + text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())
