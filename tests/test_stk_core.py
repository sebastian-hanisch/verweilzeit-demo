"""Die kopierten Kernmodule der Stapelplanung: feste Referenzwerte der Stapelplanung, Handfälle, Invarianten, unabhängige Nachrechnung."""

import pytest

import vwz_stk_core as K
from helpers import occupancy, reference_pure_moves


def test_reference_values_of_the_stapelplanung_demo():
    """Seed 7, 6 x 5, 80 %, 120 Container, Schätzfehler 0,25 (stapelplanung-demo, Commit c9f2e8d): Niedrigster Stapel 94, Bestfit 65 Umstapelungen."""
    inst = K.generate_instance(6, 5, 0.8, 120, 7)
    assert inst.capacity == 20 and inst.n_events == 240
    assert K.run_rule(inst, "niedrigster_stapel", 0.25).moves == 94
    assert K.run_rule(inst, "bestfit", 0.25).moves == 65
    assert K.run_rule(inst, "bestfit", 0.0).moves < 65                     # exakte Kenntnis hilft
    assert K.run_rule(inst, "bestfit", 0.0).retrievals == 120


def test_capacity_for():
    assert K.capacity_for(6, 5, 0.8) == 20 and K.capacity_for(6, 5, 1.0) == 25 and K.capacity_for(4, 4, 0.6) == 7 and K.capacity_for(3, 3, 0.4) == 2
    assert K.capacity_for(2, 1, 0.1) == 1                                   # nie unter 1
    assert K.capacity_for(4, 3, 0.75) == 7 and K.capacity_for(4, 3, 0.7) == 6            # 6,75 wird gerundet (nicht abgeschnitten), 6,3 auf 6


def test_generate_instance_structure_and_errors():
    inst = K.generate_instance(6, 5, 0.8, 50, 3)
    assert sorted(inst.arrival) == sorted(inst.departure) == list(range(50)) and inst.n_events == 100
    assert max(occupancy(inst.events)) <= inst.capacity and occupancy(inst.events)[-1] == 0
    assert all(inst.arrival[c] < inst.departure[c] for c in range(50))
    assert inst.mean_dwell == pytest.approx(sum(inst.departure[c] - inst.arrival[c] for c in range(50)) / 50)
    assert K.generate_instance(6, 5, 0.8, 50, 3) == inst                    # deterministisch
    for args in ((1, 5, 0.8, 10, 0), (6, 0, 0.8, 10, 0), (6, 5, 0.0, 10, 0), (6, 5, 1.01, 10, 0), (6, 5, 0.8, 0, 0)):
        with pytest.raises(ValueError):
            K.generate_instance(*args)
    K.generate_instance(2, 1, 1.0, 1, 0)                                    # kleinster gültiger Fall


def test_estimate_departures_scales_the_same_random_numbers():
    inst = K.generate_instance(6, 5, 0.8, 60, 5)
    e0, e1, e2 = (K.estimate_departures(inst, s) for s in (0.0, 0.5, 1.0))
    assert e0 == {c: float(d) for c, d in inst.departure.items()}
    for c in range(60):
        assert e2[c] - e0[c] == pytest.approx(2 * (e1[c] - e0[c]))
    z = [e1[c] - e0[c] for c in range(60)]
    assert (sum(x * x for x in z) / 60) ** 0.5 == pytest.approx(0.5 * inst.mean_dwell, rel=0.3)      # Streuung = sigma mal mittlere Standzeit (in Ereignisschritten)
    with pytest.raises(ValueError):
        K.estimate_departures(inst, -0.1)
    assert K.estimate_departures(inst, 0.5, noise_seed=1) != e1 and K.estimate_departures(inst, 0.5, noise_seed=1) == K.estimate_departures(inst, 0.5, noise_seed=1)


def test_bestfit_hand_cases():
    est = {0: 5.0, 1: 8.0, 2: 8.0}
    assert K.bestfit([[0], [1], []], 2, 6.0, est) == 1                      # knappste Passung: oben 8 >= 6 (Stapel 0 mit 5 blockiert)
    assert K.bestfit([[0], [1], []], 2, 9.0, est) == 2                      # nichts passt: der leere Stapel (unendlich)
    assert K.bestfit([[0], [1]], 2, 9.0, est) == 1                          # alle blockieren: der mit dem größten obersten
    assert K.bestfit([[1], [2]], 2, 6.0, est) == 0                          # Gleichstand der obersten: der kleinere Index
    assert K.bestfit([[0], [1], [2]], 2, 8.0, est) == 1                     # oben 8 >= 8 zählt als passend
    assert K.bestfit([[0, 1], [2], []], 2, 1.0, est) == 1                   # Stapel 0 ist voll: kommt nicht in Frage
    assert K.bestfit([[1], [2]], 2, 6.0, est, exclude=0) == 1               # ausgeschlossener Stapel
    assert K.bestfit([[1], []], 2, 8.0, est) == 0                           # oben 8 >= 8 passt und schlägt den leeren Stapel (unendlich)
    assert K.bestfit([[0], [3]], 2, 9.0, {0: 5.0, 3: 5.0}) == 0             # alle blockieren mit gleichem obersten: der kleinere Index
    assert K.bestfit([[0], [3], [4]], 2, 9.0, {0: 5.0, 3: 7.0, 4: 7.0}) == 1
    with pytest.raises(ValueError):
        K.bestfit([[0, 1]], 2, 1.0, est)


def test_niedrigster_stapel_hand_cases():
    est = {}
    assert K.niedrigster_stapel([[0, 1], [2], [3]], 3, 0.0, est) == 1       # wenigste Container, kleinster Index bei Gleichstand
    assert K.niedrigster_stapel([[0, 1, 2], [3, 4], [5, 6]], 3, 0.0, est) == 1
    assert K.niedrigster_stapel([[0], [1], []], 3, 0.0, est, exclude=2) == 0
    assert K.niedrigster_stapel([[0, 1], [2, 3], []], 2, 0.0, est) == 2     # volle Stapel zählen nicht
    with pytest.raises(ValueError):
        K.niedrigster_stapel([[0, 1], [2, 3]], 2, 0.0, est)


def test_run_counts_hand_case():
    """Drei Container mit aufsteigender Abfahrt (A0 A1 A2 D0 D1 D2), Schätzung = wahre Abfahrt."""
    events = tuple([("A", 0), ("A", 1), ("A", 2), ("D", 0), ("D", 1), ("D", 2)])
    arrival, departure = {0: 0, 1: 1, 2: 2}, {0: 3, 1: 4, 2: 5}
    est = {0: 3.0, 1: 4.0, 2: 5.0}
    three = K.Instance(3, 3, 1.0, 3, 0, 3, events, arrival, departure, 3.0)
    r = K.run(three, est, K.bestfit, record=True)                            # drei Stapel: jeder bekommt einen Container, nichts blockiert
    assert r.moves == 0 and r.retrievals == 3 and r.cumulative == (0,) * 6 and len(r.steps) == 6
    two = K.Instance(2, 3, 1.0, 3, 0, 3, events, arrival, departure, 3.0)
    r = K.run(two, est, K.bestfit)                                           # zwei Stapel: der dritte muss auf einen früher abfahrenden (kleinster Schaden), einmal umstapeln
    assert r.moves == 1
    # Ausgleich beim Einlagern: 0 -> Stapel 0, 1 -> Stapel 1, 2 -> Stapel 0 (auf 0): die Abholung von 0 stapelt 2 auf den Stapel von 1 um, die Abholung von 1 muss 2 noch einmal umstapeln
    r = K.run(two, est, K.niedrigster_stapel, record=True)
    assert r.moves == 2 and r.retrievals_with_move == 2 and r.steps[3].moved == (2,) and r.steps[4].moved == (2,) and r.max_stack_height == 2
    assert r.share_retrievals_with_move == pytest.approx(2 / 3) and r.moves_per_container == pytest.approx(2 / 3)
    assert K.run(two, est, K.niedrigster_stapel).steps == ()                 # ohne record keine Zustände


def test_max_stack_height_counts_relocations_too():
    """Vier Container auf zwei Stapel (Ausgleich): bei der Abholung von 0 landet der Blocker auf dem Stapel von 1 und 3, der dadurch dreifach hoch wird; beim Einlagern waren es höchstens zwei."""
    events = tuple([("A", 0), ("A", 1), ("A", 2), ("A", 3), ("D", 0), ("D", 1), ("D", 2), ("D", 3)])
    inst = K.Instance(2, 3, 1.0, 4, 0, 4, events, {c: c for c in range(4)}, {c: 4 + c for c in range(4)}, 4.0)
    r = K.run(inst, {c: float(c) for c in range(4)}, K.niedrigster_stapel, record=True)
    assert max(len(s) for step in r.steps[:4] for s in step.stacks) == 2 and r.steps[4].stacks == ((), (1, 3, 2)) and r.max_stack_height == 3


def test_result_ratios_are_zero_without_retrievals():
    r = K.Result(0, 0, 0, 0, (), ())
    assert r.moves_per_container == 0.0 and r.share_retrievals_with_move == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_run_invariants_on_random_instances(seed):
    inst = K.generate_instance(5, 4, 0.9, 80, seed)
    r = K.run_rule(inst, "bestfit", 0.5, record=True)
    n_present = 0
    for step, (kind, c) in zip(r.steps, inst.events):
        n_present += 1 if kind == "A" else -1
        assert sum(len(s) for s in step.stacks) == n_present and all(len(s) <= inst.max_height for s in step.stacks)
        assert step.kind == kind and step.container == c
    assert r.cumulative[-1] == r.moves == r.steps[-1].moves_so_far and all(a <= b for a, b in zip(r.cumulative, r.cumulative[1:]))
    assert r.steps[-1].stacks == ((),) * 5 and r.max_stack_height <= 4


@pytest.mark.parametrize("seed", range(8))
def test_lowest_stack_against_an_independent_simulation(seed):
    inst = K.generate_instance(6, 5, 0.8, 100, seed)
    got = K.run(inst, {c: 0.0 for c in range(100)}, K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel).moves
    assert got == reference_pure_moves(inst)


def test_relocation_rule_defaults_to_bestfit_and_can_be_overridden():
    inst = K.generate_instance(6, 5, 0.8, 100, 2)
    est = K.estimate_departures(inst, 0.25)
    a = K.run(inst, est, K.niedrigster_stapel)
    b = K.run(inst, est, K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel)
    assert K.RELOCATION_RULE is K.bestfit and a.moves != b.moves
    assert K.run(inst, est, K.niedrigster_stapel, reloc_rule=K.bestfit).moves == a.moves
