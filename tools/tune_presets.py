"""Preset-Abstimmung per Sweep: trägt die Geschichte jedes Presets im MITTEL über viele Blöcke, an der Stichprobe des Kernabschnitts, über mehrere Trainingshistorien und an dem einen Block, den das Preset zeigt?

Aufruf (im Projektordner): ./venv/Scripts/python.exe tools/tune_presets.py <modus>
  population   Grundgesamtheit (Seeds 100-299, Trainings-Seed des Presets): Kriterien aller Presets und Kennzahlen
  sample       Stichprobe des Kernabschnitts (Seeds 0-59): Urteil je Preset
  stability    Grundgesamtheit über 20 Trainingshistorien (Seeds 5-24): Vorzeichen der Differenz gelernt minus Ausgleich, dazu der Anteil der Historien, an denen der gezeigte Block die Geschichte trägt
  seeds        je Block-Seed 100..299: welche Presets tragen an diesem Block, Abstand zum Median der Differenzen; nennt die besten gemeinsamen Seeds

Grundsätze (aus den Hafen-Demos): den Seed nicht nach dem schönsten Einzelfall wählen, sondern nahe am MEDIAN; der Preset-Seed liegt außerhalb der Kernabschnitt-Stichprobe (Seeds 0-59) und in der
Grundgesamtheit (100-299) NICHT als Kriterium; alle Presets teilen sich EINEN Block-Seed; Kriterien an der Grundgesamtheit UND am gezeigten Block, weil das Urteil am einzelnen Block mit der Trainingshistorie
schwankt. Deterministisch, kein Löser: die Ergebnisse hängen nicht vom Rechner ab."""
import statistics
import sys

sys.path.insert(0, ".")
import vwz_constants as C
import vwz_evaluation as E
import vwz_stories as ST

NAMES = list(C.PRESETS)
POP_SEED0, POP_N = 100, 200
HISTORIES = range(5, 25)
RULES3 = ST.POPULATION_RULES


def params(name, train_seed=None):
    p = E.params_from_preset(C.PRESETS[name])
    return p if train_seed is None else E.Params(*p.key[:-1], train_seed)


def population(name, train_seed=None):
    return E.sample(params(name, train_seed), n=POP_N, seed0=POP_SEED0, rules=RULES3, with_quality=False)


def moves_at(rows, seed):
    """Ganze Umstapelungen der drei Regeln am Block `seed` der Grundgesamtheit."""
    r = rows[seed - POP_SEED0]
    n = C.PRESETS[NAMES[0]]["n_containers"]
    return {k: round(r.m[k] * n) for k in RULES3}


def cmd_population():
    for name in NAMES:
        rows = population(name).rows
        print(f"\n### {name}")
        for ok, text in ST.criteria(name, rows):
            print(("  OK   " if ok else "  FAIL ") + text)
        v = E.verdict(rows, C.RULE_Q70)
        print(f"  Kennzahlen: {({k: round(x, 3) for k, x in ST.key_values(name, rows).items()})}; gelernt minus Ausgleich {v.diff:+.3f} +- {v.se:.3f}, Anteil besser {E.distribution(rows, C.RULE_Q70).better:.2f}")


def cmd_sample():
    for name in NAMES:
        s = E.sample(params(name), rules=RULES3)
        v = E.verdict(s.rows, C.RULE_Q70)
        print(f"{name:18s} Stichprobe (Seeds 0-59): {v.diff:+.3f} +- {v.se:.3f} -> {v.kind}; Paarfehler {s.quality[C.RULE_Q70].pair_err:.3f}, MAE {s.quality[C.RULE_Q70].mae:.3f}")


def cmd_stability():
    seed = C.PRESETS[NAMES[0]]["seed"]
    for name in NAMES:
        diffs, holds_block, holds_pop = [], 0, 0
        for h in HISTORIES:
            rows = population(name, h).rows
            diffs.append(E.paired(rows, C.RULE_Q70)[0])
            holds_block += ST.holds(name, moves_at(rows, seed))
            holds_pop += all(ok for ok, _ in ST.criteria(name, rows))
        print(f"{name:18s} Grundgesamtheit über {len(diffs)} Historien: {min(diffs):+.3f} bis {max(diffs):+.3f} (Vorzeichen gleich: {all(d > 0 for d in diffs) or all(d < 0 for d in diffs)}); "
              f"Kriterien der Grundgesamtheit halten an {holds_pop} von {len(diffs)}, am Block (Seed {seed}) an {holds_block} von {len(diffs)}")


def cmd_seeds():
    pops = {n: population(n).rows for n in NAMES}
    diffs = {n: [pops[n][i].m[C.RULE_Q70] - pops[n][i].m[C.RULE_LOWEST] for i in range(POP_N)] for n in NAMES}
    med = {n: statistics.median(diffs[n]) for n in NAMES}
    sd = {n: statistics.pstdev(diffs[n]) + 1e-9 for n in NAMES}
    seeds = range(POP_SEED0, POP_SEED0 + POP_N)
    ok = {n: [s for s in seeds if ST.holds(n, moves_at(pops[n], s))] for n in NAMES}
    for n in NAMES:
        print(f"{n}: trägt an {len(ok[n])} von {POP_N} Blöcken")
    allgood = [s for s in seeds if all(s in ok[n] for n in NAMES)]
    score = lambda s: sum(abs(diffs[n][s - POP_SEED0] - med[n]) / sd[n] for n in NAMES)          # noqa: E731
    best = sorted(allgood, key=score)
    print(f"\nalle fünf tragen an {len(allgood)} von {POP_N} Blöcken; typischste:")
    for s in best[:6]:
        print(f"  seed {s} | Abstand zum Median {score(s):.2f} | " + ", ".join(f"{n}: {tuple(moves_at(pops[n], s).values())}" for n in NAMES))
    shown = C.PRESETS[NAMES[0]]["seed"]
    if shown in allgood:
        print(f"\ngezeigter Seed {shown}: Rang {best.index(shown) + 1} von {len(best)}")
    print("\nTypischkeit des gezeigten Blocks (10. bis 90. Perzentil der Grundgesamtheit je Kennzahl):")
    for n, k in ST.TYPICAL:
        vals = sorted(E.values(pops[n], k))
        lo, hi = vals[int(0.1 * len(vals))], vals[int(0.9 * len(vals)) - 1]
        x = pops[n][shown - POP_SEED0].m[k]
        print(f"  {n:18s} {k:24s} {x:.3f} in [{lo:.3f}, {hi:.3f}]: {lo <= x <= hi}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "population"
    {"population": cmd_population, "sample": cmd_sample, "stability": cmd_stability, "seeds": cmd_seeds}.get(mode, lambda: sys.exit(__doc__))()
