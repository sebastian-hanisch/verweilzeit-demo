"""Testhilfen: unabhängige, bewusst einfache Nachrechnungen (teilen keinen Code mit vwz_evaluation und vwz_stk_core)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def occupancy(events):
    """Belegung nach jedem Ereignis."""
    occ, out = 0, []
    for kind, _ in events:
        occ += 1 if kind == "A" else -1
        out.append(occ)
    return out


def naive_pair_error(t_arr, t_dep, est):
    """(falsch, Paare) mit einer Doppelschleife über alle Paare gleichzeitig anwesender Container."""
    n = len(t_dep)
    wrong = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t_arr[i] < t_dep[j] and t_arr[j] < t_dep[i]:
                total += 1
                if (est[i] - est[j]) * (t_dep[i] - t_dep[j]) < 0 or (est[i] == est[j]):
                    wrong += 1
    return wrong, total


def reference_pure_moves(instance):
    """Ausgleich ohne Prognose, unabhängig gebaut: Listen von Containern, beim Abholen den Stapel durchsuchen, Blocker auf den niedrigsten anderen Stapel mit Platz (kleinster Index bei Gleichstand)."""
    stacks = [[] for _ in range(instance.n_stacks)]
    H = instance.max_height
    moves = 0

    def lowest(exclude=None):
        best = None
        for i, s in enumerate(stacks):
            if i == exclude or len(s) >= H:
                continue
            if best is None or len(s) < len(stacks[best]):
                best = i
        return best

    for kind, c in instance.events:
        if kind == "A":
            stacks[lowest()].append(c)
        else:
            x = next(i for i, s in enumerate(stacks) if c in s)
            while stacks[x][-1] != c:
                b = stacks[x].pop()
                stacks[lowest(exclude=x)].append(b)
                moves += 1
            stacks[x].pop()
    return moves
