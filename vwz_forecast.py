"""Klebstoff: Lerner -> Prognose der Verweilzeit (Median, Mittelwert, Quantil 0,7, Streuung) -> Schätzung der Abfahrt -> Regeln der Stapelplanung.

Die Schätzung der Abfahrt ist Ankunftszeit + Prognose (in Tagen); die Regeln (Bestfit, Niedrigster Stapel) sind unverändert die der Stapelplanung (`vwz_stk_core`). Die unsicherheitsbewusste Regel
bekommt je Container die prognostizierte log-Verteilung (Median mu, Streuung s) über eine `Est`-Schätzung (float-Unterklasse: Regeln, die nur die Zahl brauchen, sehen eine gewöhnliche Zahl).

Modelle werden je (Ankündigung, Trainingsmenge, Trainings-Seed) einmal trainiert und im Prozess gemerkt (`functools.lru_cache`, rein deterministisch)."""

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import vwz_constants as C
import vwz_generator as G
import vwz_models as M
import vwz_stk_core as K


class Est(float):
    """Geschätzte Abfahrtszeit (Tage) mit Zusatzinformation für unsicherheitsbewusste Regeln: Ankunftszeit t, log-Median mu, log-Streuung s."""
    __slots__ = ("t", "mu", "s")


def make_est(t_arr, mu, s):
    """Est je Container aus Ankunftszeit und vorhergesagter log-Verteilung der Verweilzeit (Median-Schätzung)."""
    out = {}
    for c in range(len(t_arr)):
        e = Est(t_arr[c] + math.exp(mu[c]))
        e.t, e.mu, e.s = float(t_arr[c]), float(mu[c]), float(s[c])
        out[c] = e
    return out


def est_plain(values):
    return {c: float(v) for c, v in enumerate(values)}


def true_est(sample):
    """Hellseher: die wahre Abfahrtszeit."""
    return est_plain(sample.t_dep)


def noisy_est(sample, sigma, seed):
    """Wahre Abfahrt (Tage) + Gauß-Rauschen sigma * mittlere Verweilzeit (Tage): das Rauschen der Stapelplanung auf der Zeitachse."""
    rng = random.Random(seed)
    scale = sigma * sample.mean_dwell_days
    return {c: float(sample.t_dep[c]) + rng.gauss(0.0, 1.0) * scale for c in range(len(sample.t_dep))}


def _phi(x):
    return 0.5 * (1 + np.array([math.erf(v) for v in x / math.sqrt(2)]))


def make_aware(nodes=12):
    """Unsicherheitsbewusst: minimiert P(eigener Container fährt NACH dem obersten). Verteilungen aus (mu, s) je Container: lognormal, T = t + exp(mu + s z).
    P(T_top < T_own) per Gauß-Hermite über z_own. Rundung und Schwellen der Stapelplanung (2 Stellen, 0,3) unverändert."""
    xs, ws = np.polynomial.hermite_e.hermegauss(nodes)
    ws = ws / ws.sum()

    def aware(stacks, H, e_own, est, exclude=None):
        best = None
        t_own = np.array(e_own.t + np.exp(e_own.mu + e_own.s * xs))          # mögliche eigene Abfahrten (Knoten)
        for i in K.candidates(stacks, H, exclude):
            s = stacks[i]
            if s:
                top = est[s[-1]]
                x = t_own - top.t
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = (np.log(np.where(x > 0, x, 1e-300)) - top.mu) / max(top.s, 1e-9)
                F = np.where(x > 0, _phi(z), 0.0)                             # P(T_top <= t_own)
                p = float(np.dot(ws, F))
                topv = float(top)
            else:
                p, topv = 0.0, K.INF
            key = (round(p, 2), topv if p < 0.3 else len(s), i)
            if best is None or key < best[0]:
                best = (key, i)
        return best[1]
    return aware


# ---------------------------------------------------------------------------------------------------
# Lerner
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Forecast:
    """Prognose der Verweilzeit für eine Reihe von Containern (Tage; mu und s: log-Verteilung)."""
    median: np.ndarray
    mean: np.ndarray
    q70: np.ndarray
    mu: np.ndarray
    s: np.ndarray


@lru_cache(maxsize=96)
def history(nu, n_train, train_seed):
    """Trainingsdaten (X, Verweilzeit in Tagen)."""
    return G.make_history(n_train, train_seed, nu=nu)


@lru_cache(maxsize=96)
def _boost(nu, n_train, train_seed, kind):
    """Ein Boosting-Modell auf log-Verweilzeit: 'q10', 'q50', 'q70', 'q90' (Quantilverlust) oder 'l2' (Mittelwert)."""
    X, y = history(nu, n_train, train_seed)
    ly = np.log(y)
    if kind == "l2":
        model = M.GBT("l2", n_rounds=C.BOOST_ROUNDS, depth=C.BOOST_DEPTH, lr=C.BOOST_LR, min_leaf=C.BOOST_MIN_LEAF, max_bins=C.BOOST_BINS)
    else:
        model = M.GBT("quantile", int(kind[1:]) / 100, n_rounds=C.BOOST_ROUNDS, depth=C.BOOST_DEPTH, lr=C.BOOST_LR, min_leaf=C.BOOST_MIN_LEAF, max_bins=C.BOOST_BINS)
    return model.fit(X, ly)


@lru_cache(maxsize=96)
def _linear(nu, n_train, train_seed):
    X, y = history(nu, n_train, train_seed)
    return M.LogLinear().fit(X, y)


@lru_cache(maxsize=96)
def _group(n_train, train_seed):
    X, y = history(None, n_train, train_seed)             # die Verweilzeiten hängen nicht von der Ankündigung ab, das Gruppenmittel nutzt nur die Art
    return M.GroupMedian("mean").fit(X, y)


def forecast(learner, nu, n_train, train_seed, X):
    """Volle Prognose (Median, Mittelwert, Quantil 0,7, log-Verteilung) des Lerners für die Merkmale X."""
    if learner == C.LEARNER_LINEAR:
        ll = _linear(nu, n_train, train_seed)
        mu = ll.mu(X)
        s = np.full(len(X), ll.s)
        return Forecast(np.exp(mu), np.exp(mu + ll.s ** 2 / 2), np.exp(mu + ll.s * C.Z_QUANTILE), mu, s)
    if learner == C.LEARNER_BOOSTING:
        q10, q50, q70, q90 = (_boost(nu, n_train, train_seed, k).predict(X) for k in ("q10", "q50", "q70", "q90"))
        s = np.maximum((q90 - q10) / C.S_BAND_DIVISOR, C.S_FLOOR)
        l2 = _boost(nu, n_train, train_seed, "l2").predict(X)
        return Forecast(np.exp(q50), np.exp(l2 + 0.5 * s ** 2), np.exp(q70), q50, s)
    raise ValueError(f"Unbekannter Lerner: {learner!r}")


def forecast_curve(learner, nu, n_train, train_seed, X):
    """Die Prognose in Tagen, die in der Lernkurve zählt: vorsichtiges Quantil 0,7 (ein Modell statt fünf); beim Gruppenmittel der Gruppenmittelwert (Vergleichslinie)."""
    if learner == C.LEARNER_LINEAR:
        ll = _linear(nu, n_train, train_seed)
        return np.exp(ll.mu(X) + ll.s * C.Z_QUANTILE)
    if learner == C.LEARNER_BOOSTING:
        return np.exp(_boost(nu, n_train, train_seed, "q70").predict(X))
    if learner == C.LEARNER_GROUP:
        return _group(n_train, train_seed).predict(X)
    raise ValueError(f"Unbekannter Lerner: {learner!r}")
