"""Kernmodule der Stapelplanung, KOPIERT (nicht importiert): Ereignisfolge im Block, Einlagerungsregeln, Ablauf mit Umstapeln.

Herkunft: `stapelplanung-demo` (stk_scenario.py, stk_rules.py, stk_simulation.py, Stand Commit c9f2e8d), auf das hier Nötige gekürzt und in einer Datei zusammengefasst. Jede Demo ist ein eigenständiges
Repo; deshalb liegt der Code hier und nicht als Import. Damit die Kopie nicht unbemerkt abweicht, pinnt `tests/test_stk_core.py` feste Referenzwerte der Stapelplanung (Seed 7, 6 x 5, 80 %, 120 Container,
Schätzfehler 0,25: Niedrigster Stapel 94, Bestfit 65 Umstapelungen) und ein paar handgerechnete Fälle.

Ein Block besteht aus n_stacks Stapeln der Höhe max_height. Container kommen und gehen in einer Ereignisfolge ("A" = Ankunft, "D" = Abholung). Jede Regel bekommt den Blockzustand und entscheidet,
in welchen Stapel ein Container kommt: rule(stacks, max_height, e_own, est, exclude=None) -> Stapelindex (e_own = geschätzte Abfahrt des Containers, est = Schätzung aller Container).
Beim Abholen werden nur die Container über dem Ziel umgestapelt, jeder Hub zählt als eine Umstapelung; wohin ein Blocker kommt, entscheidet `reloc_rule` (Standard: Bestfit)."""

import random
from dataclasses import dataclass

INF = float("inf")

# Wahrscheinlichkeit, dass der nächste Schritt eine Ankunft ist (nur für generate_instance, den Vergleichsmodus der Stapelplanung).
P_ARRIVAL = 0.55
# Das Rauschen der Stapelplanung kommt aus einem eigenen Zufallsstrom, abgeleitet aus dem Ereignis-Seed.
NOISE_SEED_MULTIPLIER = 13
NOISE_SEED_OFFSET = 5


@dataclass(frozen=True)
class Instance:
    n_stacks: int
    max_height: int
    fill: float            # Anteil von (n_stacks - 1) * max_height, 0 < fill <= 1
    n_containers: int
    seed: int
    capacity: int          # höchste gleichzeitige Belegung
    events: tuple          # ((kind, container), ...), kind in {"A", "D"}
    arrival: dict          # container -> Index des Ankunfts-Ereignisses
    departure: dict        # container -> Index des Abholungs-Ereignisses (wahre Abfahrt in Schritten)
    mean_dwell: float      # mittlere Standzeit in Ereignisschritten

    @property
    def n_events(self):
        return len(self.events)


def capacity_for(n_stacks, max_height, fill):
    """Höchste gleichzeitige Belegung. Bis (n_stacks - 1) * max_height + 1 Container finden beim Abholen immer Platz zum Umstapeln."""
    return max(1, round(fill * (n_stacks - 1) * max_height))


def generate_instance(n_stacks, max_height, fill, n_containers, seed):
    """Ereignisfolge der Stapelplanung (Standzeit ohne Gedächtnis: ein zufälliger anwesender Container geht). Hier nur als Vergleichsmaßstab für den Modus 'ohne Gedächtnis' des Generators."""
    if n_stacks < 2:
        raise ValueError("Mindestens 2 Stapel nötig (sonst gibt es kein Umstapeln).")
    if max_height < 1:
        raise ValueError("Stapelhöhe muss mindestens 1 sein.")
    if not 0 < fill <= 1:
        raise ValueError("Füllgrad muss in (0, 1] liegen.")
    if n_containers < 1:
        raise ValueError("Mindestens 1 Container nötig.")

    rng = random.Random(seed)
    cap = capacity_for(n_stacks, max_height, fill)
    events, present, arrived = [], [], 0
    while arrived < n_containers or present:
        may_arrive = arrived < n_containers and (not present or (len(present) < cap and rng.random() < P_ARRIVAL))
        if may_arrive:
            events.append(("A", arrived))
            present.append(arrived)
            arrived += 1
        else:
            container = present.pop(rng.randrange(len(present)))
            events.append(("D", container))

    arrival = {c: t for t, (kind, c) in enumerate(events) if kind == "A"}
    departure = {c: t for t, (kind, c) in enumerate(events) if kind == "D"}
    mean_dwell = sum(departure[c] - arrival[c] for c in departure) / len(departure)
    return Instance(n_stacks=n_stacks, max_height=max_height, fill=fill, n_containers=n_containers, seed=seed, capacity=cap,
                    events=tuple(events), arrival=arrival, departure=departure, mean_dwell=mean_dwell)


def estimate_departures(instance, sigma, noise_seed=None):
    """Geschätzte Abfahrt (in Ereignisschritten) je Container: wahre Abfahrt + Gauß-Rauschen. sigma = Schätzfehler als Bruchteil der mittleren Standzeit."""
    if sigma < 0:
        raise ValueError("sigma darf nicht negativ sein.")
    rng = random.Random(instance.seed * NOISE_SEED_MULTIPLIER + NOISE_SEED_OFFSET if noise_seed is None else noise_seed)
    scale = sigma * instance.mean_dwell
    return {c: d + rng.gauss(0.0, 1.0) * scale for c, d in instance.departure.items()}


# ---------------------------------------------------------------------------------------------------
# Einlagerungsregeln
# ---------------------------------------------------------------------------------------------------
def candidates(stacks, max_height, exclude):
    """Stapel mit freiem Platz in Indexreihenfolge (kanonisch: keine Scheinvorteile durch Gleichstände)."""
    cand = [i for i, s in enumerate(stacks) if i != exclude and len(s) < max_height]
    if not cand:
        raise ValueError("Kein Stapel mit freiem Platz: Belegung liegt über der zulässigen Grenze.")
    return cand


def bestfit(stacks, max_height, e_own, est, exclude=None):
    """Nichts Früheres blockieren, möglichst eng: Stapel mit dem kleinsten obersten ê >= eigenem ê (leere Stapel zählen als unendlich, werden also zuletzt genommen).
    Blockiert jeder Kandidat, der Stapel mit dem größten obersten ê (kleinster Schaden)."""
    cand = candidates(stacks, max_height, exclude)
    good = bad = None
    for i in cand:
        s = stacks[i]
        top = est[s[-1]] if s else INF
        if top >= e_own:
            if good is None or top < good[0]:
                good = (top, i)
        elif bad is None or top > bad[0]:
            bad = (top, i)
    return (good or bad)[1]


def niedrigster_stapel(stacks, max_height, e_own, est, exclude=None):
    """Ausgleich: Stapel mit den wenigsten Containern (bei Gleichstand der kleinste Index). Ignoriert die Schätzung."""
    cand = candidates(stacks, max_height, exclude)
    return min(cand, key=lambda i: (len(stacks[i]), i))


RULES = {"niedrigster_stapel": niedrigster_stapel, "bestfit": bestfit}

# Für ALLE Regeln entscheidet dieselbe Regel, wohin ein Blocker beim Abholen umgestapelt wird (Konvention der Stapelplanung).
RELOCATION_RULE = bestfit


# ---------------------------------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """Zustand nach einem Ereignis (nur wenn record=True)."""
    kind: str               # "A" Ankunft, "D" Abholung
    container: int
    stack: int              # Ankunft: gewählter Stapel; Abholung: Stapel, aus dem abgeholt wurde
    moved: tuple            # Abholung: umgestapelte Container (oberster zuerst), sonst ()
    stacks: tuple           # Blockzustand, Tupel von Tupeln (unten nach oben)
    moves_so_far: int


@dataclass(frozen=True)
class Result:
    moves: int
    retrievals: int
    retrievals_with_move: int
    max_stack_height: int
    cumulative: tuple       # kumulierte Umstapelungen nach jedem Ereignis
    steps: tuple            # leer, wenn nicht aufgezeichnet

    @property
    def moves_per_container(self):
        return self.moves / self.retrievals if self.retrievals else 0.0

    @property
    def share_retrievals_with_move(self):
        return self.retrievals_with_move / self.retrievals if self.retrievals else 0.0


def run(instance, estimates, place_rule, reloc_rule=RELOCATION_RULE, record=False):
    """Simuliert die Ereignisfolge der Instanz mit der Einlagerungsregel place_rule und der Umstaplregel reloc_rule."""
    H = instance.max_height
    stacks = [[] for _ in range(instance.n_stacks)]
    where = {}
    moves = retrievals = with_move = max_height_seen = 0
    cumulative, steps = [], []

    for kind, c in instance.events:
        moved = []
        if kind == "A":
            i = place_rule(stacks, H, estimates[c], estimates)
            stacks[i].append(c)
            where[c] = i
            max_height_seen = max(max_height_seen, len(stacks[i]))
            stack_used = i
        else:
            x = where[c]
            s = stacks[x]
            while s[-1] != c:
                b = s.pop()
                j = reloc_rule(stacks, H, estimates[b], estimates, exclude=x)
                stacks[j].append(b)
                where[b] = j
                moved.append(b)
                max_height_seen = max(max_height_seen, len(stacks[j]))
            s.pop()
            del where[c]
            retrievals += 1
            if moved:
                with_move += 1
            moves += len(moved)
            stack_used = x
        cumulative.append(moves)
        if record:
            steps.append(Step(kind, c, stack_used, tuple(moved), tuple(tuple(s) for s in stacks), moves))

    return Result(moves, retrievals, with_move, max_height_seen, tuple(cumulative), tuple(steps))


def run_rule(instance, rule_key, sigma, noise_seed=None, record=False):
    """Bequemer Einstieg der Stapelplanung: schätzt die Abfahrten mit Fehler sigma (Bruchteil der mittleren Standzeit) und simuliert die Regel ('bestfit' oder 'niedrigster_stapel')."""
    est = estimate_departures(instance, sigma, noise_seed)
    return run(instance, est, RULES[rule_key], record=record)
