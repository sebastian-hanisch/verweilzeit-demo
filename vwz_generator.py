"""Synthetischer Containerstrom mit Merkmalen und Verweilzeit (Annahmen, kein Echtdatenbezug). Bitgleich zu `hafen-planung/messreihe_verweilzeit/gen.py` (Test pinnt feste Werte).

Merkmale, die zum ANKUNFTSZEITPUNKT bekannt sind -> Verweilzeit (in Tagen, lognormal, heteroskedastisch):
  art      0 Import, 1 Export, 2 Umschlag, 3 Leer
  kunde    0..K-1  (Kundeneffekt auf die log-Verweilzeit)
  zoll     1 = Zollhalt bei Import (verdoppelt etwa die Verweilzeit)
  wochenende  Ankunft Sa/So (Import und Leer bleiben länger)
  vorlauf  Tage bis zur geplanten Schiffsabfahrt (Export/Umschlag), sonst 0
  gewicht  reines Rauschmerkmal (Wirkung null)
  angekuendigt  von Spediteur oder Reederei angekündigte Verweilzeit in Tagen = wahre Verweilzeit * exp(nu * z), z ~ N(0, 1) unabhängig; nu=None: keine Ankündigung (Spalte 0).
           Der Regler nu ("Verlässlichkeit der Ankündigung") ändert die wahren Verweilzeiten NICHT (gemeinsame Zufallszahlen), nur das Signal.
Die Zielgröße (Verweilzeit) wird erst NACH den Merkmalen gezogen; Merkmale enthalten keine Zukunftsinformation.

Ereignisfolge: Ankünfte als Poisson-Strom (Rate lam), wahre Abfahrt = Ankunft + Verweilzeit, Belegungsgrenze cap wie in der Stapelplanung (ist der Block voll, verzögert sich die Ankunft bis zur
nächsten Abfahrt). Ergebnis enthält eine `Instance` der Stapelplanung, damit deren Regeln und Ablauf unverändert laufen."""

import heapq
import math
from dataclasses import dataclass

import numpy as np

import vwz_constants as C
import vwz_stk_core as K

ART_P, K_KUNDEN, U_SD, U_SCALE, S_BASE, P_ZOLL = C.ART_P, C.K_KUNDEN, C.U_SD, C.U_SCALE, C.S_BASE, C.P_ZOLL


def kunden_effekte(k_seed=C.KUNDEN_SEED):
    return np.random.default_rng(k_seed).normal(0.0, U_SD, K_KUNDEN)


UK = kunden_effekte()


def log_median_and_s(art, kunde, zoll, we, vorlauf, noise_scale=1.0):
    """Analytische bedingte Verteilung: log-Verweilzeit ~ N(mu, s^2). Vektorisiert. Das ist der 'Orakel'-Wert."""
    art = np.asarray(art)
    u = UK[kunde] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    mu = np.log(med) + u
    s = np.take(S_BASE, art) * noise_scale
    return mu, s


@dataclass
class Sample:
    inst: object            # vwz_stk_core.Instance
    X: np.ndarray           # (n, N_FEATURES) Merkmale in Ankunftsreihenfolge = Container-ID
    dwell: np.ndarray       # Verweilzeit in Tagen
    t_arr: np.ndarray       # Ankunftszeit (Tage)
    t_dep: np.ndarray       # wahre Abfahrtszeit (Tage)
    mu: np.ndarray          # Orakel: log-Median
    s: np.ndarray           # Orakel: log-Streuung
    mean_dwell_days: float


def draw_features(rng, n):
    """Nur für Trainingsdaten: unabhängige Historie ohne Blockbeschränkung (Ankunftszeit gleichverteilt über viele Wochen)."""
    art = rng.choice(4, size=n, p=ART_P)
    kunde = rng.integers(0, K_KUNDEN, n)
    zoll = ((rng.random(n) < P_ZOLL) & (art == 0)).astype(int)
    tday = rng.random(n) * 7000.0
    we = ((np.floor(tday).astype(int) % 7) >= 5).astype(int)
    vorlauf = np.where(art == 1, rng.uniform(1, 10, n), np.where(art == 2, rng.uniform(0.5, 6, n), 0.0))
    gewicht = rng.normal(0, 1, n)
    return np.column_stack([art, kunde, zoll, we, vorlauf, gewicht, np.zeros(n)]).astype(float)


def make_history(n, seed, noise_scale=1.0, nu=None):
    """Trainingsdaten: n abgeschlossene Container aus einer eigenen Historie (eigener Seed-Bereich): X, Verweilzeit."""
    rng = np.random.default_rng(C.HISTORY_SEED_OFFSET + seed)
    X = draw_features(rng, n)
    mu, s = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4], noise_scale)
    z = rng.standard_normal(n)
    za = rng.standard_normal(n)
    d = np.exp(mu + s * z)
    if nu is not None:
        X[:, C.COL_ANNOUNCED] = d * np.exp(nu * za)
    return X, d


def bayes_oracle(mu, s, ang, nu):
    """Exakte bedingte log-Verteilung der Verweilzeit gegeben Merkmale UND Ankündigung (Orakel für nu > 0)."""
    if nu is None:
        return mu, s
    if nu <= 0:
        return np.log(ang), np.zeros_like(mu)
    p1, p2 = 1 / s ** 2, 1 / nu ** 2
    return (mu * p1 + np.log(ang) * p2) / (p1 + p2), np.sqrt(1 / (p1 + p2))


_MEAN_CACHE = {}


def marginal_mean_dwell(noise_scale=1.0):
    """Mittlere Verweilzeit des Stroms in Tagen (Stichprobe von 400 000 Containern, deterministisch); legt zusammen mit rho die Ankunftsrate fest."""
    if noise_scale not in _MEAN_CACHE:
        _, d = make_history(C.MEAN_DWELL_N, C.MEAN_DWELL_SEED, noise_scale)
        _MEAN_CACHE[noise_scale] = float(d.mean())
    return _MEAN_CACHE[noise_scale]


def generate(n_stacks, max_height, fill, n_containers, seed, noise_scale=1.0, rho=C.RHO, memoryless=False, nu=None):
    """Ein Block-Lauf. rho = Ankunftsrate * mittlere Verweilzeit / cap (mittlere Belegung ohne Grenze relativ zur Grenze)."""
    if not 0 < fill <= 1:
        raise ValueError("Füllgrad muss in (0, 1] liegen.")
    if n_containers < 1:
        raise ValueError("Mindestens 1 Container nötig.")
    cap = K.capacity_for(n_stacks, max_height, fill)
    lam = rho * cap / marginal_mean_dwell(noise_scale)
    # Gemeinsame Zufallszahlen je Container (unabhängig von noise_scale, rho und nu, damit Varianten vergleichbar sind); drei getrennte Ströme, damit die ersten k Container
    # von n_containers NICHT abhängen (Test: keine Zukunftsinformation).
    U = np.random.default_rng([seed, 1]).random((n_containers, 6))
    Z = np.random.default_rng([seed, 2]).standard_normal((n_containers, 3))
    E = np.random.default_rng([seed, 3]).exponential(1.0, n_containers)
    cum_art = np.cumsum(ART_P)

    heap, events = [], []
    t_prev = 0.0
    feats, dwell, t_arr, mus, ss = [], [], [], [], []
    for k in range(n_containers):
        t = t_prev + E[k] / lam
        while heap and heap[0][0] <= t:
            _, c = heapq.heappop(heap)
            events.append(("D", c))
        if len(heap) >= cap:
            td, c = heapq.heappop(heap)
            events.append(("D", c))
            t = max(t, td)
        u = U[k]
        art = min(int(np.searchsorted(cum_art, u[0])), 3)
        kunde = int(u[1] * K_KUNDEN)
        zoll = int(art == 0 and u[2] < P_ZOLL)
        we = int((int(t) % 7) >= 5)
        vorlauf = 1 + 9 * u[3] if art == 1 else (0.5 + 5.5 * u[3] if art == 2 else 0.0)
        gew = float(Z[k, 1])
        mu, s = log_median_and_s(np.array(art), kunde, zoll, we, vorlauf, noise_scale)
        d = float(np.exp(mu + s * Z[k, 0]))
        if memoryless:
            d = marginal_mean_dwell(noise_scale) * float(-np.log(1 - u[4]))
        ang = d * math.exp(nu * Z[k, 2]) if nu is not None else 0.0
        feats.append((art, kunde, zoll, we, vorlauf, gew, ang))
        dwell.append(d)
        t_arr.append(t)
        mus.append(float(mu))
        ss.append(float(s))
        events.append(("A", k))
        heapq.heappush(heap, (t + d, k))
        t_prev = t
    while heap:
        _, c = heapq.heappop(heap)
        events.append(("D", c))
    arrival = {c: i for i, (kd, c) in enumerate(events) if kd == "A"}
    departure = {c: i for i, (kd, c) in enumerate(events) if kd == "D"}
    mean_dwell_steps = sum(departure[c] - arrival[c] for c in departure) / len(departure)
    inst = K.Instance(n_stacks=n_stacks, max_height=max_height, fill=fill, n_containers=n_containers, seed=seed, capacity=cap,
                      events=tuple(events), arrival=arrival, departure=departure, mean_dwell=mean_dwell_steps)
    dwell = np.array(dwell)
    t_arr = np.array(t_arr)
    return Sample(inst, np.array(feats, float), dwell, t_arr, t_arr + dwell, np.array(mus), np.array(ss), float(dwell.mean()))
