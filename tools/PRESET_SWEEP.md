# Preset-Abstimmung (AP 5)

Werkzeug: `tools/tune_presets.py` (Modi `population`, `sample`, `stability`, `seeds`); Kriterien in `vwz_stories.py`, Abnahme in `tests/test_preset_stories.py` (echte Daten; die Stabilität über 20 Historien
als `slow`) und `tests/test_stories.py` (künstliche Werte an den Schwellen). Deterministisch, kein Löser: die Ergebnisse hängen nicht vom Rechner ab.

Messbasis: Block 6 × 5, 300 Container, Boosting mit 100 Runden und Quantil 0,7, Trainings-Seed 5; Grundgesamtheit = 200 Blöcke mit den Seeds 100 bis 299, Stichprobe des Kernabschnitts = 60 Blöcke mit den Seeds 0 bis 59.
Die Zahlen reproduzieren `hafen-planung/messreihe_verweilzeit/presets_final300.json` auf die dritte Stelle (Test `test_the_population_reproduces_the_messreihe`), der Block 268 bitgleich.

## Grundgesamtheit (200 Blöcke, Differenz = gelernt minus Ausgleich je Container, ± Standardfehler)

| Preset | Einstellung | Ausgleich | gelernt | Differenz | Anteil der Blöcke mit weniger | Hellseher | Urteil |
|---|---|---|---|---|---|---|---|
| Nur Standard | keine Ankündigung, 1000 Trainingscontainer | 0,860 | 0,926 | **+0,066 ± 0,007** | 21 % | 0,252 | kippt |
| Mit Ankündigung | ν 0,6, 1000 | 0,860 | 0,745 | **−0,115 ± 0,006** | 90 % | 0,252 | Vorteil |
| Zu wenig Daten | ν 0,6, 30 | 0,860 | 0,976 | **+0,116 ± 0,007** | 14 % | 0,252 | kippt |
| Sehr verlässlich | ν 0,3, 1000 | 0,860 | 0,598 | **−0,262 ± 0,005** | 100 % | 0,252 | Vorteil |
| Voller Block | ν 0,6, 1000, Füllgrad 100 % | 1,136 | 0,993 | **−0,143 ± 0,007** | 92 % | 0,496 | Vorteil |

Stichprobe des Kernabschnitts (Seeds 0 bis 59): Nur Standard +0,058 ± 0,010 (kippt), Mit Ankündigung −0,121 ± 0,011 (Vorteil), Zu wenig Daten +0,111 ± 0,011 (kippt), Sehr verlässlich −0,264 ± 0,010 (Vorteil),
Voller Block −0,154 ± 0,013 (Vorteil); Paarfehler 30 / 21 / 32 / 15 / 21 %.

## Kriterien (Schwellen mit Abstand, ganze Zahlen am Block)

| Preset | Grundgesamtheit | gezeigter Block (Seed 268) | gemessen am Block (Ausgleich / gelernt / Hellseher) |
|---|---|---|---|
| Nur Standard | Differenz ≥ +0,03, Urteil "kippt" | gelernt ≥ Ausgleich + 5 | 238 / 252 / 79 |
| Mit Ankündigung | ≤ −0,06, Urteil "Vorteil", Hellseher ≤ 40 % des Ausgleichs | gelernt ≤ Ausgleich − 8, Hellseher ≤ 40 % | 238 / 222 / 79 |
| Zu wenig Daten | ≥ +0,06, "kippt" | gelernt ≥ Ausgleich + 20 | 238 / 281 / 79 |
| Sehr verlässlich | ≤ −0,18, "Vorteil" | gelernt ≤ 80 % des Ausgleichs | 238 / 160 / 79 |
| Voller Block | ≤ −0,08, "Vorteil", Ausgleich ≥ 1,0 je Container | Ausgleich ≥ 300 und gelernt ≤ Ausgleich − 25 | 355 / 310 / 148 |

## Trainingshistorie: die Grundgesamtheit hält ihr Vorzeichen, der einzelne Block nicht immer

Über 20 Trainingshistorien (Trainings-Seeds 5 bis 24), jeweils dieselben 200 Blöcke:

| Preset | Differenz der Grundgesamtheit | Vorzeichen gleich | Kriterien der Grundgesamtheit halten | am Block 268 halten die Kriterien |
|---|---|---|---|---|
| Nur Standard | +0,038 bis +0,080 | ja | 20 von 20 | 19 von 20 |
| Mit Ankündigung | −0,138 bis −0,092 | ja | 20 von 20 | 19 von 20 |
| Zu wenig Daten | +0,019 bis +0,172 | ja | **13 von 20** | 15 von 20 |
| Sehr verlässlich | −0,273 bis −0,224 | ja | 20 von 20 | 19 von 20 |
| Voller Block | −0,163 bis −0,116 | ja | 20 von 20 | 19 von 20 |

**Befund.** Bei "Zu wenig Daten" (30 Trainingscontainer) ist das Ergebnis eine Ziehung: in 7 von 20 Historien liegt die Differenz der Grundgesamtheit unter der Schwelle +0,06 (kleinster Wert +0,019), das Vorzeichen bleibt
aber überall gleich. Die Schwelle steht bei +0,06, weil der Trainings-Seed 5 des Presets +0,116 liefert; die Geschichte ("schlechter als der Ausgleich") gilt in jeder Ziehung, ihre Größe nicht. Der Test
`test_population_keeps_its_sign_over_20_training_histories` prüft deshalb das Vorzeichen in allen 20 Ziehungen und die Schwellen nur teilweise (mindestens 10 beziehungsweise 18 von 20). Der Trainings-Seed-Regler macht das sichtbar.

## Gewählt

- Eine gemeinsame Block-Nummer für alle Presets: **Seed 268**. Von 200 Blöcken (Seeds 100 bis 299) tragen 50 alle fünf Geschichten am Block (Kriterien oben); 268 hat unter ihnen den kleinsten Abstand zum Median der
  Differenzen (1,04; danach 138 mit 1,18, 119 mit 1,32, 209, 191, 294). Der Plan nannte 56 Blöcke und dieselbe Nummer 268; er zählte mit den Kriterien des Suchskripts der Messreihe, die etwas andere Schwellen hatten
  (Zahl der Blöcke deshalb 50 statt 56, der Seed derselbe).
- Jede Kennzahl des gezeigten Blocks (Ausgleich, gelernt, Hellseher je Container) liegt zwischen dem 10. und 90. Perzentil der Grundgesamtheit (Test `test_the_shown_block_is_typical_for_every_key_measure`), der Block liegt
  außerhalb der Kernabschnitt-Stichprobe (Seeds ab 60).
- Trainings-Seed fest 5, damit die Geschichte am gezeigten Block reproduzierbar ist; der Regler zeigt, wie stark kleine Trainingsmengen schwanken.
