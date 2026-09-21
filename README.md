# Verweilzeit lernen statt annehmen – Streamlit-Demo

Interaktive Fall-Demo zu **Predict-then-Optimize im Containerblock**: Ein Container kommt an, und man weiß nur ungefähr, wann er abgeholt wird. In der Stapelplanung-Demo war diese Schätzung ein Regler (wahre Abfahrt plus
Gauß-Rauschen); hier wird die **Verweilzeit aus Merkmalen gelernt** (synthetische Daten, lineare Regression und selbst gebautes Gradient Boosting, nur `numpy`) und in die Einlagerungsregel gesteckt. Die Demo beantwortet:
**Wie gut muss die Prognose sein, wie viele Trainingsdaten braucht sie, und ab wann ist sie besser als der einfache Ausgleich (Niedrigster Stapel, ganz ohne Prognose)?**

Teil des Portfolios für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Zusatz zur Hafen-Linie (setzt auf der `stapelplanung-demo` auf: Regeln und Ablauf sind deren Kern, hier kopiert).
Framing: Optimierung (Umstapeln vermeiden) und Maschinelles Lernen (Prognose) sowohl eigenständig als auch kombiniert.

## Warum dieses Problem, und was die Messung am Aufhänger geändert hat

Naheliegend wäre „aus Merkmalen lernen schlägt den einfachen Ausgleich, und mehr Daten machen es besser“. Gemessen trägt das **nicht allgemein**. Mit Standardmerkmalen (Art, Kunde, Zoll, Wochenende, Vorlauf bis Schiff) schlägt
selbst das **Orakel**, das die wahre bedingte Verteilung kennt, den Ausgleich bei 300 Containern nicht (+0,043 ± 0,007 Umstapelungen je Container mit Median-Prognose, −0,004 ± 0,006 mit vorsichtigem Quantil 0,7); die gelernte
Prognose liegt bei +0,061 ± 0,007 und auch mit 10 000 Trainingscontainern bei −0,001. Erst eine verlässliche Vorabinformation (angekündigte Verweilzeit, etwa der Abholtermin) trägt: −0,124 ± 0,006 bei mittlerer, −0,263 ± 0,005 bei
guter Verlässlichkeit; der Hellseher liegt bei −0,607 ± 0,004 (Ausgleich 0,862 je Container). Tragend ist deshalb: **Eine Prognose ist nur so viel wert wie ihr Signal, und ihr Wert hängt am Anteil richtig geordneter Paare, nicht
am mittleren Fehler.**

## Modell

Ankünfte als Poisson-Strom, Belegungsgrenze wie in der Stapelplanung (ein voller Block verzögert die Ankunft); Merkmale zur Ankunft: Containerart (Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %), Kunde (12 mit Effekt), Zollhalt
(nur Import), Wochenende, Vorlauf bis zur Schiffsabfahrt (Export, Umschlag), ein reines Rauschmerkmal. Die **Verweilzeit** wird danach gezogen: lognormal und heteroskedastisch (Mittel rund 5,6 Tage). Optional gibt es die
**Ankündigung** = wahre Verweilzeit × exp(ν z); der Regler „Verlässlichkeit“ ändert die wahren Verweilzeiten **nicht** (gemeinsame Zufallszahlen), nur das Signal. Trainingsdaten sind eine unabhängige Historie mit derselben Verteilung.
Alles ist **erfunden** (kein Echtdatenbezug); die Größenordnung des Nutzens hängt an der angenommenen unerklärten Streuung (siehe Grenzen).

## Methodik – vier Regeln plus Einordnung

Alle Regeln laufen auf derselben Ereignisfolge (dieselben Merkmale und Zufallszahlen); Bezug aller Deltas ist **Niedrigster Stapel ohne jede Prognose** (Einlagern und Umstapeln nach niedrigstem Stapel).

- **🚛 Niedrigster Stapel:** Stapel mit den wenigsten Containern, keine Prognose (Bezug).
- **🎯 Bestfit mit Median:** Bestfit auf Ankunft plus Median der prognostizierten Verweilzeit (die MAE-optimale Prognose).
- **🛡️ Bestfit vorsichtig (Quantil 0,7):** Bestfit auf Ankunft plus Quantil 0,7 (je Container ein eigener Aufschlag beim Boosting).
- **🎲 Unsicherheitsbewusst:** die Regel der Stapelplanung auf der prognostizierten Verteilung (Gauß-Hermite), unverändert und nicht neu abgestimmt.
- Im Vergleich zusätzlich: Bestfit mit Mittelwert, **Hellseher** (wahre Abfahrt, Obergrenze, kennt keine künftigen Ankünfte) und Niedrigster Stapel, der mit derselben Prognose umstapelt (Konvention der Stapelplanung).

Lerner: lineare Regression auf die log-Verweilzeit; Gradient Boosting (Histogramm-Bäume der Tiefe 3, 100 Runden, Lernrate 0,1, Quantilverlust 0,1 / 0,5 / 0,7 / 0,9 und L2 auf der log-Verweilzeit). Das Gruppenmittel nach Art erscheint nur als
Vergleichslinie in der Lernkurve. Der **Kernabschnitt** rechnet ohne Knopf (60 Blöcke, Seeds 0 bis 59, gecacht): Paarfehler-Kurve mit dem Gauß-Rauschen der Stapelplanung und Kipppunkt, Lernkurve über der Trainingsmenge (Band über
vier Trainings-Seeds bei kleinen Mengen), Median gegen Mittelwert gegen Quantil, Urteil in drei Zuständen mit der Verteilung dazu.

## Befunde (gemessen, keine Behauptungen)

300 Container, Block 6 × 5, Füllgrad 80 %, 200 Blöcke (Seeds 0 bis 199), gepaarte Differenzen ± Standardfehler; die tragenden Zahlen mit dem Code dieser Demo nachgemessen (`tests/test_messreihe.py`, `tests/test_preset_stories.py`, `tools/PRESET_SWEEP.md`; Angaben aus den Robustheitsreihen bei 120 Containern stammen aus der Messreihe).

| Frage | Befund |
|---|---|
| **Stimmt die Rechnung?** | Der Generator ist bitgleich zur Messreihe (`hafen-planung/messreihe_verweilzeit/gen.py`, feste Referenzwerte im Test), der Hellseher **Zustand für Zustand** gleich `run_rule(σ = 0)` der Stapelplanung, die kopierten Kernmodule reproduzieren deren Referenzwerte (Seed 7, σ 0,25: 94 und 65 Umstapelungen), der Modus „ohne Gedächtnis“ die Stapelplanung bei gleicher Belegung (±12 %), das Boosting stimmt in der Messreihe mit scikit-learn auf 0,2 bis 1,1 % (Pinball) überein (hier als optionaler Test, wenn installiert). |
| **Was bringt die gelernte Prognose?** | Ausgleich 0,862 je Container; Boosting (Quantil 0,7, 1000 Trainingscontainer) mit Ankündigung ν 0,6: **−0,124 ± 0,006**, ν 0,3: **−0,263 ± 0,005**; nur Standardmerkmale **+0,061 ± 0,007** (schlechter). Mit 100 statt 150 Runden: −0,120 / −0,261 / +0,060. |
| **Wie gut muss sie sein?** | Bestfit mit Gauß-Rauschen schlägt den Ausgleich bis **σ ≈ 0,78 Standzeiten = 27 % falsch geordnete Paare** (MAE 0,62); bei 120 Containern 0,71. Der Nutzen korreliert mit dem Paarfehler (0,998), nicht mit dem MAE (0,805 mit den Gauß-Punkten). |
| **Median oder Quantil?** | Die MAE-optimale Median-Prognose ist die **schlechteste**: Quantil 0,7 gegen Median −0,043 ± 0,006 (ν 0,6), obwohl der MAE steigt; nur bei Füllgrad ≥ 80 % (in 8 von 8 Einstellungen der Messreihe), bei 60 % kein Unterschied. |
| **Wie viele Daten?** | Boosting bei ν 0,6: N = 30 **+0,103**, 100 +0,027, 300 −0,073, 1000 −0,124, 10 000 −0,159; lineare Regression ab N = 100 flach; das Gruppenmittel nach Art (+0,18 bis +0,24) schlägt den Ausgleich nie. |
| **Hilft die unsicherheitsbewusste Regel?** | Meist nicht: mit den wahren Verteilungen +0,030 (ν 0,6) bis +0,105 (ν 0,3) gegen den Median (schlechter, je besser die Prognose); bei 6 × 5 und Füllgrad ≥ 80 % schlechter als das Quantil, bei 4 × 4 und 60 % besser. |
| **Presets** | Eine gemeinsame Block-Nummer (Seed 268) und ein Trainings-Seed (5) für alle fünf; Kriterien an der Grundgesamtheit (200 Blöcke) UND am gezeigten Block; jede Kennzahl des Blocks zwischen dem 10. und 90. Perzentil. |

## Ehrliche Grenzen

- Merkmale, Verweilzeit-Verteilung und Ankündigung sind **Annahmen** (die Ankündigung mit lognormalem Fehler, ohne Ausreißer und fehlende Werte).
- **Die unerklärte Streuung ist die entscheidende Annahme** (Messreihe, 120 Container, 200 Blöcke; nachgemessen): Streuung × 0,5 / 0,75 / 1,0 / 1,5 lässt die exakte Kenntnis der Standardmerkmale −0,272 / −0,145 / −0,015 / **+0,180** je Container
  gegen den Ausgleich sparen. Ob sich Lernen aus Standardmerkmalen lohnt, hängt vollständig daran.
- Die Ergebnisse hängen an der **Lauflänge**: bei 120 Containern (Anlauf und Auslauf sind leicht) ist „Nur Standard“ gleichauf (+0,017 ± 0,010), bei 300 kippt es (+0,066 ± 0,007). Standard deshalb 300; Robustheit (Blöcke und
  Füllgrade), Streuung und Aktualisierung liegen in der Messreihe nur bei 120 Containern vor.
- Training und Test aus derselben Verteilung (kein Saisonwechsel; in der Messreihe kostet eine um 30 % zu kurze Prognose etwa die Hälfte des Nutzens); Prognose nur zur Ankunft (die Aktualisierung mit dem Alter bringt nach der Messreihe nichts);
  Abholreihenfolge allein nach der Verweilzeit; ein Block, keine Kranwege; der Hellseher kennt keine künftigen Ankünfte.
- Das Boosting ist **nicht abgestimmt**; „vorsichtiges Quantil 0,7“ ist aus der Messreihe gewählt (0,8 bis 0,9 sind bei ν 0,6 noch etwas besser, nicht in der Robustheitsreihe geprüft).
- Was eine bessere Prognose kostet (Daten, Erhebung, Pflege), ist Sache des Betreibers und **kein Geldbetrag**; alle Zahlen sind **Größenordnungen aus einer Simulation, keine Messung an einem Terminal**.

## Befunde / Korrekturen gegenüber Messreihe und Plan

- **Gruppenmittel nach Art nicht wählbar.** Der Plan listete es als dritten Lerner. Es ignoriert die Ankündigung, der Regler „Verlässlichkeit“ wäre dort ein toter Regler (er änderte nichts). Es erscheint deshalb nur als
  Vergleichslinie in der Lernkurve; wählbar sind lineare Regression und Boosting, und jeder Regler wirkt in jeder Einstellung.
- **Rundenzahl.** Die Presets der Messreihe (und die Trainingszeit) beruhen auf 100 Boosting-Runden, ihre Tabellen auf 150. Die App nutzt 100; die Kennzahlen der Befunde-Tabelle mit 150 Runden nachgemessen
  (−0,124 / −0,263 / +0,061) stimmen, mit 100 Runden liegen sie bei −0,120 / −0,261 / +0,060.
- **Lernkurve mit Quantil 0,7 für beide Lerner.** Die Messreihe zeigte die lineare Regression mit dem Mittelwert; die App zeigt beide mit derselben Prognose (Quantil 0,7), damit Lerner und nicht Prognoseart verglichen werden;
  das Gruppenmittel mit dem Mittelwert je Art. Die Werte der linearen Regression weichen deshalb von der Tabelle der Messreihe ab.
- **Kernabschnitt mit 60 statt 200 Blöcken** (Rechenzeit ohne Knopf): Kipppunkt σ 0,79 statt 0,78, äquivalentes Rauschen 0,48 wie in der Messreihe.
- **„Zu wenig Daten“ ist eine Ziehung.** In 7 von 20 Trainingshistorien liegt die Differenz der Grundgesamtheit unter der Preset-Schwelle +0,06 (kleinster Wert +0,019); das Vorzeichen (schlechter als der Ausgleich) gilt in
  allen 20. Die Tests prüfen deshalb das Vorzeichen und die Schwellen nur teilweise (`tools/PRESET_SWEEP.md`).
- **Preset-Seed 268 bestätigt, Zählung anders.** Mit den Kriterien des Plans (Abschnitt 7) tragen 50 statt 56 von 200 Blöcken alle fünf Geschichten; die Messreihe zählte mit einem Suchskript mit anderen Schwellen. 268 hat weiter den
  kleinsten Abstand zum Median.
- **Meldung und Urteil an der Stichprobe.** Die bedingte Meldung der Hauptansicht stützt ihr Urteil auf die 60 Blöcke der Stichprobe und nennt den gezeigten Block daneben; ein Urteil am einzelnen Block wäre mit der
  Trainingshistorie umgeschlagen (Messreihe, Korrektur 6).
- **Bezug der Ersparnis** ist Niedrigster Stapel ohne Prognose (nicht der Niedrigste Stapel mit derselben Prognose beim Umstapeln); dieser steht als eigene Zeile im Vergleich (im Vergleich schon das Umstapeln allein mit Prognose
  senkt die Umstapelungen: 0,698 gegen 0,862 in der Messreihe).

## Tests

`python -m pytest tests/ -v` – 284 Tests plus 1 übersprungener (die Gegenprobe gegen scikit-learn, nur mit installiertem scikit-learn), rund 10 Minuten unter Windows und 12 Minuten in einem Linux-Container mit frischer Installation
(Python 3.12, neueste Pakete; `tools/demo_linux_check.py`); `-m "not slow"` lässt die Nachmessung der Messreihe und die Stabilität über 20 Trainingshistorien aus (rund eine Minute Rechnung, ohne AppTests). Zusammensetzung:

- **Rechnung:** Kernmodule der Stapelplanung gegen deren feste Referenzwerte, Handfälle, Invarianten und eine **unabhängige Simulation** (Ausgleich); Generator gegen feste Werte der Messreihe, Ereignisfolge, **keine Zukunftsinformation**,
  Ankündigung, Trainingsdaten getrennt von den Blöcken; Hellseher **bitgleich** zu `run_rule(σ = 0)` in jedem Zustand jedes Ereignisses, Grenzfall ν = 0 gleich Hellseher; Bayes-Orakel gegen eine Monte-Carlo-Bedingung.
- **Lerner:** lineare Regression gegen `np.linalg.lstsq`, Boosting an Handfällen, Determinismus, **Leckage-Sonde** (vertauschte Zielgrößen lernen nichts), Gegenprobe gegen scikit-learn (nur wenn installiert).
- **Auswertung:** Paarfehler gegen eine Doppelschleife, Urteil an der Schwelle, Verteilung, Gauß-Kurve gegen die Messreihe, Kipppunkt, äquivalentes Rauschen, Lernkurve, Meldung in allen Zuständen.
- **Presets:** Geschichte am gezeigten Block, im Mittel über 200 Blöcke, typisch je Kennzahl, alle Kriterien einzeln an ihren Schwellen mit künstlichen Werten (die Schwellen stehen fest im Test), Stabilität über 20 Historien.
- **Nachmessung** der Messreihe im Toleranzband: −0,124 / −0,263, +0,061, Orakel +0,043, Quantil gegen Median, Kipppunkt, Trainingsmenge.
- **CI-robust:** keine Bänder um einen einzelnen chaotischen Lauf und keine exakten Extreme rauschender Größen; das Boosting ist deterministisch (feste Seeds, deterministische Gleichstände).
- **PDF, Figuren, End-to-End (AppTest):** Sonderzeichen mit den genauen Zeichen, Inhalt Zelle für Zelle, feste Achsen; Skelett und Footer, jedes Preset, Permalink, alle Regler an Min und Max, jeder Lerner mit und ohne
  Ankündigung, Meldung in allen Zuständen, Vergleichstabelle, Texte.

Zusätzlich wurde jedes Modul mit **eingebauten Fehlern** geprüft (`tools/mutation_check.py`, 155 Mutanten, parallel, mit Zeitgrenze je Mutant; ohne AppTests und ohne die langsamen Nachmessungen). Der erste Lauf fand 134
von 155; von den 20 Überlebenden waren **zehn echte Lücken der Tests**, sie sind geschlossen und erneut geprüft: die Rundung der Belegungsgrenze (`round` gegen Abschneiden), der Gleichstand beim Bestfit gegen einen leeren Stapel, die Stapelhöhe
beim Umstapeln, die Formel der exponentiellen Verweilzeit im Modus „ohne Gedächtnis“, der Startwert des L2-Boostings, die Klassenbildung bei genau so vielen verschiedenen Werten wie Klassen, die Risikoschwelle 0,3 der unsicherheitsbewussten
Regel, die Blockgröße in der Stichprobe, der Seitenumbruch im PDF und die Untergrenze der Boosting-Streuung; ein weiterer Überlebender war ein falsch geschriebener Mutant. **Zehn Überlebende sind gleichwertig** und bleiben:
fünf Kriterien der Grundgesamtheit der Presets genau auf einer Gleitkommagrenze (`>=` gegen `>` bei den Schwellen ±0,03 / ±0,06 / ±0,18 / ±0,08; die ganzzahligen Schwellen am Block sind exakt geprüft),
`t = max(t, td)` im Generator (nach dem Leeren des Haufens ist die nächste Abfahrt immer später, das `max` ändert nichts), die Ankündigung im Verlauf des Gruppenmittels (die Verweilzeiten hängen nicht von ν ab), der Vorzeichenwechsel `<` gegen `<=`
in der Kipppunkt-Suche (ein exakter Nullwert wäre schon im vorigen Intervall gefunden), die Schrittweite `> 1` gegen `> 0` beim Einrasten des Permalinks (bei Schrittweite 1 ist das Einrasten die Identität) und `y > F` gegen `y >= F` im Gradienten des
Quantil-Boostings (gilt nur bei exakten Gleichständen in der ersten Runde; der Baum ist derselbe).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Sidebar, Hauptansicht, Blick in den Block, Kernabschnitt, Regelvergleich, Texte |
| `vwz_constants.py` | Modellparameter, Lerner, Regeln, Regler-Grenzen, `PRESETS`, Farben |
| `vwz_presets.py` | `SETTING_SPECS`, Permalink (Begrenzen und Einrasten), Presets, Seed-Knopf |
| `vwz_generator.py` | Strom mit Merkmalen, Verweilzeit, Ankündigung; Historie; Bayes-Orakel (bitgleich zur Messreihe) |
| `vwz_models.py` | Lerner nur mit numpy: Gruppenmittel, lineare Regression, Gradient Boosting (L2 und Quantil) |
| `vwz_forecast.py` | Lerner → Prognose (Median, Mittelwert, Quantil, Streuung) → Schätzung; unsicherheitsbewusste Regel |
| `vwz_stk_core.py` | **Kopie** der Kernmodule der Stapelplanung (Ereignisfolge, Regeln, Ablauf) mit Herkunftsvermerk |
| `vwz_evaluation.py` | Regeln je Block, Stichprobe, Gauß-Kurve, Kipppunkt, Lernkurve, Urteil, Verteilung, Meldung |
| `vwz_visualization.py` | Blockansicht, Paarfehler-Kurve, Lernkurve, Balken, Verteilung, Streudiagramm (alle Achsen fest) |
| `vwz_ui_panel.py` | Kennzahlen 2 × 2, Panel je Regel, Schrittregler |
| `vwz_pdf_export.py` | PDF-Ergebnis (`fpdf2`, Kernschrift, Sonderzeichen-Bereinigung) |
| `vwz_stories.py` | Abnahmekriterien der Presets (Quelle für Werkzeug und Tests) |
| `tools/tune_presets.py`, `tools/PRESET_SWEEP.md` | Preset-Abstimmung und ihr Bericht |
| `tools/mutation_check.py` | Fehler-Einbau-Test |
| `tests/` | siehe oben |

## Bewusst nicht umgesetzt (mögliche Erweiterungen)

- Prognose mit dem Alter aktualisieren (Messreihe: Median bringt nichts, Quantil 0,7 schadet), Saisonwechsel und Verteilungsbruch als Regler, ein Regler für die unerklärte Streuung, echte Verweilzeitdaten.
- Bündelung nach Schiff, weitere Lerner (Zufallswald, neuronale Netze), Ankündigungen mit Ausreißern oder fehlenden Werten, Abstimmung der Boosting-Parameter.
- Kopplung an die Gate- oder Kaiplatz-Demo (die Ankündigung wäre dort der gebuchte Abholtermin), Optimum mit Hellsehen und bekannten künftigen Ankünften aus der Stapelplanung.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `python -m pytest tests/ -v`. Preset-Abstimmung: `python tools/tune_presets.py population|sample|stability|seeds`. Fehler-Einbau: `python tools/mutation_check.py`.

---
