"""Der synthetische Strom: feste Referenzwerte der Messreihe (gen.py), Ereignisfolge, keine Zukunftsinformation, Ankündigung, Trainingsdaten, Bayes-Orakel, Modus 'ohne Gedächtnis'."""

import hashlib
import math

import numpy as np
import pytest

import vwz_constants as C
import vwz_generator as G
import vwz_stk_core as K
from helpers import occupancy


def event_hash(inst):
    return hashlib.sha256(",".join(f"{k}{c}" for k, c in inst.events).encode()).hexdigest()[:16]


# Referenzwerte aus hafen-planung/messreihe_verweilzeit/gen.py (rho 0,7): (Seed, nu, Stapel, Höhe, Füllgrad, Container) -> Werte
REF = {
    (268, 0.6, 6, 5, 0.8, 300): dict(cap=20, mean_dwell=27.64666666666667, hash="f74191d54387f350", dwell_sum=1748.8691191445992, t_last=121.06979740560224, x_sum=4735.957992685569,
                                     ang_sum=1947.2418914760515, art=[108, 84, 81, 27]),
    (268, None, 6, 5, 0.8, 300): dict(cap=20, mean_dwell=27.64666666666667, hash="f74191d54387f350", dwell_sum=1748.8691191445992, t_last=121.06979740560224, x_sum=2788.716101209517, ang_sum=0.0,
                                      art=[108, 84, 81, 27]),
    (0, 0.3, 4, 4, 0.6, 120): dict(cap=7, mean_dwell=9.033333333333333, hash="e3ef98710a22215b", dwell_sum=639.0786369445002, t_last=141.575529666959, x_sum=1867.3537403097866,
                                   ang_sum=678.6445931591885, art=[38, 41, 32, 9]),
    (5, 0.15, 6, 5, 1.0, 300): dict(cap=25, mean_dwell=34.46, hash="7b552d976c39ee9e", dwell_sum=1821.9283381218895, t_last=98.50465958911354, x_sum=4715.603067154968,
                                    ang_sum=1849.902257356764, art=[104, 93, 70, 33]),
}


@pytest.mark.parametrize("args", list(REF))
def test_generator_reproduces_the_messreihe(args):
    seed, nu, S, H, fill, n = args
    ref = REF[args]
    s = G.generate(S, H, fill, n, seed, nu=nu)
    assert s.inst.capacity == ref["cap"] and s.inst.mean_dwell == pytest.approx(ref["mean_dwell"], rel=1e-9) and event_hash(s.inst) == ref["hash"]
    assert s.dwell.sum() == pytest.approx(ref["dwell_sum"], rel=1e-9) and s.t_arr[-1] == pytest.approx(ref["t_last"], rel=1e-9)
    assert s.X.sum() == pytest.approx(ref["x_sum"], rel=1e-9) and s.X[:, 6].sum() == pytest.approx(ref["ang_sum"], rel=1e-9, abs=1e-12)
    assert np.bincount(s.X[:, 0].astype(int)).tolist() == ref["art"]


def test_first_containers_of_the_preset_block_pinned():
    s = G.generate(6, 5, 0.8, 300, 268, nu=0.6)
    assert s.dwell[:3] == pytest.approx([8.680527397114501, 4.497453068995579, 4.844127654575722], rel=1e-9)
    assert G.marginal_mean_dwell() == pytest.approx(5.648373754923789, rel=1e-6)          # rund 5,6 Tage


def test_history_reproduces_the_messreihe():
    X, y = G.make_history(1000, 5, nu=0.6)
    assert X.sum() == pytest.approx(16210.571697300453, rel=1e-9) and y.sum() == pytest.approx(5784.074939390839, rel=1e-9)
    assert X[1].tolist() == pytest.approx([2.0, 9.0, 0.0, 1.0, 3.6135858121260886, 0.9174133507147779, 1.9317459235269439], rel=1e-9)
    assert y[:3] == pytest.approx([1.9183866778419896, 4.444922419737849, 1.1760476132762563], rel=1e-9)
    X0, y0 = G.make_history(1000, 5, nu=None)
    assert np.array_equal(y0, y) and np.array_equal(X0[:, :6], X[:, :6]) and not X0[:, 6].any() and X[:, 6].all()      # nu ändert nur die Ankündigung, nie die wahren Zeiten


@pytest.mark.parametrize("seed", range(6))
def test_event_sequence_invariants(seed):
    s = G.generate(6, 5, 0.8, 120, seed)
    ev, cap = s.inst.events, s.inst.capacity
    assert len(ev) == 240 and max(occupancy(ev)) <= cap and min(occupancy(ev)) >= 0 and occupancy(ev)[-1] == 0
    seen = set()
    for kind, c in ev:
        if kind == "A":
            assert c not in seen
            seen.add(c)
        else:
            assert c in seen                                                   # Abholung erst nach der Ankunft
    deps = [c for k, c in ev if k == "D"]
    assert all(s.t_dep[a] <= s.t_dep[b] for a, b in zip(deps, deps[1:]))         # Abholungen in Abfahrtszeit-Reihenfolge
    arrs = [c for k, c in ev if k == "A"]
    assert arrs == sorted(arrs) and all(s.t_arr[a] <= s.t_arr[b] for a, b in zip(arrs, arrs[1:]))
    assert np.allclose(s.t_dep, s.t_arr + s.dwell)


def test_full_block_delays_arrivals():
    """Ist der Block voll, verzögert sich die Ankunft bis zur nächsten Abfahrt: bei hoher Last liegt die Belegung an der Grenze."""
    busy = G.generate(6, 5, 0.8, 200, 4, rho=3.0)
    assert max(occupancy(busy.inst.events)) == busy.inst.capacity
    assert np.all(np.diff(busy.t_arr) >= 0)


def test_no_future_information():
    big = G.generate(6, 5, 0.8, 120, 11, nu=0.6)
    small = G.generate(6, 5, 0.8, 60, 11, nu=0.6)
    assert np.array_equal(big.X[:60], small.X) and np.array_equal(big.t_arr[:60], small.t_arr) and np.array_equal(big.dwell[:60], small.dwell)
    a = G.generate(6, 5, 0.8, 120, 5, noise_scale=1.0)
    b = G.generate(6, 5, 0.8, 120, 5, noise_scale=2.0)
    assert np.array_equal(a.X[0, :6], b.X[0, :6]) and a.dwell[0] != b.dwell[0]  # Merkmale des ersten Containers hängen nicht von der Streuung der Verweilzeit ab


def test_announcement_reliability_changes_only_the_signal():
    a = G.generate(6, 5, 0.8, 150, 9, nu=0.6)
    b = G.generate(6, 5, 0.8, 150, 9, nu=0.3)
    n = G.generate(6, 5, 0.8, 150, 9, nu=None)
    assert a.inst.events == b.inst.events == n.inst.events and np.array_equal(a.dwell, b.dwell) and np.array_equal(a.t_arr, b.t_arr)
    assert np.array_equal(a.X[:, :6], b.X[:, :6]) and not np.array_equal(a.X[:, 6], b.X[:, 6]) and not n.X[:, 6].any()
    z = np.log(a.X[:, 6] / a.dwell) / 0.6
    assert np.allclose(np.log(b.X[:, 6] / b.dwell) / 0.3, z)                     # dieselbe Zufallszahl z, nur mit nu skaliert


def test_announcement_error_has_the_stated_scale():
    """ln(Ankündigung / Verweilzeit) ist normal mit Streuung nu: Streuung und Mittel über viele Container."""
    ratios = np.concatenate([np.log(s.X[:, 6] / s.dwell) for s in (G.generate(6, 5, 0.8, 300, sd, nu=0.6) for sd in range(20))])
    assert ratios.mean() == pytest.approx(0.0, abs=0.03) and ratios.std() == pytest.approx(0.6, abs=0.03)


def test_features_are_within_their_ranges():
    s = G.generate(6, 5, 0.8, 300, 3, nu=0.6)
    art, kunde, zoll, we, vor = (s.X[:, i] for i in range(5))
    assert set(art) <= {0, 1, 2, 3} and kunde.min() >= 0 and kunde.max() < C.K_KUNDEN and set(zoll) <= {0, 1} and set(we) <= {0, 1}
    assert np.all(zoll[art != 0] == 0)                                          # Zollhalt nur bei Import
    assert np.all(vor[(art == 0) | (art == 3)] == 0) and np.all((vor[art == 1] >= 1) & (vor[art == 1] <= 10)) and np.all((vor[art == 2] >= 0.5) & (vor[art == 2] <= 6))
    assert np.array_equal(we, ((np.floor(s.t_arr).astype(int) % 7) >= 5).astype(float))    # Wochenende der Ankunft
    assert s.mu.shape == s.s.shape == (300,) and np.all(s.s > 0)


def test_log_median_and_s_hand_values():
    mu, s = G.log_median_and_s(np.array([0, 0, 1, 2, 3]), np.zeros(5, int), np.array([0, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 1]), np.array([0.0, 0.0, 2.0, 4.0, 0.0]))
    u = G.UK[0]
    expected = [math.log(3.5) + u, math.log(3.5) + 0.7 + u, math.log(0.6 + 0.85 * 2.0) + 0.3 * u, math.log(0.3 + 0.8 * 4.0) + 0.3 * u, math.log(7.0) + 0.3 + u]
    assert mu == pytest.approx(expected) and s == pytest.approx([0.55, 0.55, 0.25, 0.35, 0.90])
    assert G.log_median_and_s(np.array([0]), np.zeros(1, int), np.zeros(1), np.zeros(1), np.zeros(1), noise_scale=2.0)[1] == pytest.approx([1.1])


def test_dwell_distribution_matches_the_stated_model():
    """Über viele Container: mittlere Verweilzeit rund 5,6 Tage, die Streuung der log-Verweilzeit wächst von Export über Import zu Leer, Zollhalt verdoppelt etwa."""
    X, y = G.make_history(200_000, 1)
    art, zoll = X[:, 0], X[:, 2]
    assert y.mean() == pytest.approx(5.65, abs=0.15)
    ls = lambda a: float(np.std(np.log(y[art == a])))                            # noqa: E731
    assert ls(1) < ls(0) < ls(3) and ls(3) > 0.85
    assert np.median(y[(art == 0) & (zoll == 1)]) / np.median(y[(art == 0) & (zoll == 0)]) == pytest.approx(2.0, rel=0.1)


def test_training_history_is_separate_from_the_test_blocks():
    Xh, _ = G.make_history(3000, 5, nu=0.6)
    Xt = np.vstack([G.generate(6, 5, 0.8, 120, sd, nu=0.6).X for sd in range(30)])
    key = lambda X: {tuple(np.round(r[[1, 4, 5]], 9)) for r in X}                # noqa: E731
    assert not (key(Xh) & key(Xt))
    assert not np.array_equal(G.make_history(50, 1)[1], G.make_history(50, 2)[1]) and np.array_equal(G.make_history(50, 1)[1], G.make_history(50, 1)[1])


def test_bayes_oracle_against_monte_carlo_conditioning():
    mu0, s0, nu = 1.2, 0.5, 0.4
    rng = np.random.default_rng(1)
    ld = mu0 + s0 * rng.standard_normal(400_000)
    lang = ld + nu * rng.standard_normal(400_000)
    mu_p, s_p = G.bayes_oracle(np.full(1, mu0), np.full(1, s0), np.exp(np.array([1.8])), nu)
    sel = np.abs(lang - 1.8) < 0.01
    assert mu_p[0] == pytest.approx(ld[sel].mean(), abs=0.02) and s_p[0] == pytest.approx(ld[sel].std(), abs=0.02)
    mu_n, s_n = G.bayes_oracle(np.full(2, mu0), np.full(2, s0), None, None)
    assert mu_n.tolist() == [mu0, mu0] and s_n.tolist() == [s0, s0]               # keine Ankündigung: Merkmale allein
    mu_e, s_e = G.bayes_oracle(np.full(2, mu0), np.full(2, s0), np.exp(np.array([0.7, 2.0])), 0.0)
    assert mu_e == pytest.approx([0.7, 2.0]) and s_e.tolist() == [0.0, 0.0]        # exakte Ankündigung


def test_memoryless_mode_reproduces_the_stapelplanung_at_equal_occupancy():
    """Modus 'ohne Gedächtnis' (exponentielle Verweilzeit) gegen den Strom der Stapelplanung bei angeglichener Belegung (Toleranz +-12 %, wie in der Messreihe)."""
    NI = 200
    mem = [G.generate(6, 5, 0.8, 120, sd, memoryless=True) for sd in range(NI)]
    org = [K.generate_instance(6, 5, 0.8, 120, sd) for sd in range(NI)]
    mid = lambda ev: float(np.mean(occupancy(ev)[60:180]))                       # noqa: E731
    assert np.mean([mid(x.inst.events) for x in mem]) == pytest.approx(np.mean([mid(x.events) for x in org]), rel=0.12)
    low_m = np.mean([K.run(x.inst, {c: float(x.t_dep[c]) for c in range(120)}, K.niedrigster_stapel).moves / 120 for x in mem])
    bf_m = np.mean([K.run(x.inst, {c: float(x.t_dep[c]) for c in range(120)}, K.bestfit).moves / 120 for x in mem])
    low_o = np.mean([K.run_rule(x, "niedrigster_stapel", 0).moves / 120 for x in org])
    bf_o = np.mean([K.run_rule(x, "bestfit", 0).moves / 120 for x in org])
    assert low_m == pytest.approx(low_o, rel=0.12) and bf_m == pytest.approx(bf_o, rel=0.12)


def test_memoryless_mode_is_pinned_to_the_messreihe():
    """Exponentielle Verweilzeit = mittlere Verweilzeit mal -ln(1 - u) (gen.py, memoryless, Seed 3, 120 Container)."""
    m = G.generate(6, 5, 0.8, 120, 3, memoryless=True)
    assert m.dwell[0] == pytest.approx(4.370935847350292, rel=1e-9) and m.dwell.sum() == pytest.approx(669.0496120458351, rel=1e-9) and m.t_arr[-1] == pytest.approx(51.9113160309309, rel=1e-9)


def test_generate_errors_and_edge_cases():
    for args in ((6, 5, 0.0, 10, 0), (6, 5, 1.5, 10, 0), (6, 5, 0.8, 0, 0)):
        with pytest.raises(ValueError):
            G.generate(*args)
    one = G.generate(6, 5, 0.8, 1, 0)
    assert one.inst.n_events == 2 and one.X.shape == (1, C.N_FEATURES)
    assert G.generate(6, 5, 1.0, 5, 0).inst.n_containers == 5


def test_kunden_effekte_are_fixed():
    assert G.UK.shape == (C.K_KUNDEN,) and np.array_equal(G.UK, G.kunden_effekte()) and not np.array_equal(G.UK, G.kunden_effekte(1))
    assert G.UK.std() == pytest.approx(0.3, abs=0.15)
