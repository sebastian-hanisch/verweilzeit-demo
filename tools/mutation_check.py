"""Fehler-Einbau-Test: baut einzelne Fehler in die Module ein und prüft, ob die Tests (ohne AppTests und ohne die langsamen Nachmessungen) sie finden.

Aufruf (im Projektordner): ./venv/Scripts/python.exe tools/mutation_check.py [Teilstring des Dateinamens] [Nummern,durch,Komma]
Jeder Mutant ersetzt genau eine Stelle (Mutanten laufen parallel, MUT_WORKERS Arbeiter, je Arbeiter eine eigene Kopie); Überlebende sind entweder gleichwertig (kein sichtbarer Unterschied) oder eine Lücke der Tests. Die Kopie liegt in einem temporären Ordner;
PYTHONDONTWRITEBYTECODE=1, damit veralteter Bytecode keine Überlebenden vortäuscht; Quelltexte als LF (Windows-Python schreibt sonst CRLF und die Zeichenketten unten finden nichts).
Ein Mutant kann in eine Endlosschleife laufen; nach TIMEOUT Sekunden gilt er als gefunden."""
import concurrent.futures
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
TIMEOUT = 240                      # Sekunden je Mutant; Endlosschleifen zählen als gefunden

MUTANTS = [
    # vwz_stk_core.py
    ("vwz_stk_core.py", "return max(1, round(fill * (n_stacks - 1) * max_height))", "return max(0, round(fill * (n_stacks - 1) * max_height))"),
    ("vwz_stk_core.py", "return max(1, round(fill * (n_stacks - 1) * max_height))", "return max(1, int(fill * (n_stacks - 1) * max_height))"),
    ("vwz_stk_core.py", "if top >= e_own:", "if top > e_own:"),
    ("vwz_stk_core.py", "if good is None or top < good[0]:", "if good is None or top <= good[0]:"),
    ("vwz_stk_core.py", "elif bad is None or top > bad[0]:", "elif bad is None or top >= bad[0]:"),
    ("vwz_stk_core.py", "top = est[s[-1]] if s else INF", "top = est[s[-1]] if s else -INF"),
    ("vwz_stk_core.py", "return min(cand, key=lambda i: (len(stacks[i]), i))", "return min(cand, key=lambda i: (len(stacks[i]), -i))"),
    ("vwz_stk_core.py", "cand = [i for i, s in enumerate(stacks) if i != exclude and len(s) < max_height]", "cand = [i for i, s in enumerate(stacks) if i != exclude and len(s) <= max_height]"),
    ("vwz_stk_core.py", "cand = [i for i, s in enumerate(stacks) if i != exclude and len(s) < max_height]", "cand = [i for i, s in enumerate(stacks) if len(s) < max_height]"),
    ("vwz_stk_core.py", "scale = sigma * instance.mean_dwell", "scale = sigma"),
    ("vwz_stk_core.py", "if sigma < 0:", "if sigma <= 0:"),
    ("vwz_stk_core.py", "            moves += len(moved)", "            moves += 1"),
    ("vwz_stk_core.py", "            if moved:\n                with_move += 1", "            if not moved:\n                with_move += 1"),
    ("vwz_stk_core.py", "max_height_seen = max(max_height_seen, len(stacks[j]))", "max_height_seen = max(max_height_seen, len(stacks[i]) if False else 0)"),
    ("vwz_stk_core.py", "j = reloc_rule(stacks, H, estimates[b], estimates, exclude=x)", "j = reloc_rule(stacks, H, estimates[b], estimates)"),
    ("vwz_stk_core.py", "if n_stacks < 2:", "if n_stacks < 1:"),
    ("vwz_stk_core.py", "P_ARRIVAL = 0.55", "P_ARRIVAL = 0.5"),
    # vwz_generator.py
    ("vwz_generator.py", "lam = rho * cap / marginal_mean_dwell(noise_scale)", "lam = rho * cap * marginal_mean_dwell(noise_scale)"),
    ("vwz_generator.py", "if len(heap) >= cap:", "if len(heap) > cap:"),
    ("vwz_generator.py", "we = int((int(t) % 7) >= 5)", "we = int((int(t) % 7) >= 6)"),
    ("vwz_generator.py", "vorlauf = 1 + 9 * u[3] if art == 1", "vorlauf = 1 + 8 * u[3] if art == 1"),
    ("vwz_generator.py", "d = float(np.exp(mu + s * Z[k, 0]))", "d = float(np.exp(mu - s * Z[k, 0]))"),
    ("vwz_generator.py", "ang = d * math.exp(nu * Z[k, 2]) if nu is not None else 0.0", "ang = d * math.exp(nu * Z[k, 2]) if nu is not None else 1.0"),
    ("vwz_generator.py", "if not 0 < fill <= 1:", "if not 0 <= fill <= 1:"),
    ("vwz_generator.py", "if n_containers < 1:", "if n_containers < 0:"),
    ("vwz_generator.py", "X[:, C.COL_ANNOUNCED] = d * np.exp(nu * za)", "X[:, C.COL_ANNOUNCED] = d * np.exp(nu * z)"),
    ("vwz_generator.py", "tday = rng.random(n) * 7000.0", "tday = rng.random(n) * 7001.0"),
    ("vwz_generator.py", "if nu <= 0:", "if nu < 0:"),
    ("vwz_generator.py", "np.sqrt(1 / (p1 + p2))", "1 / (p1 + p2)"),
    ("vwz_generator.py", "kunde = int(u[1] * K_KUNDEN)", "kunde = int(u[1] * (K_KUNDEN - 1))"),
    ("vwz_generator.py", "t = t_prev + E[k] / lam", "t = t_prev + E[k] * lam"),
    ("vwz_generator.py", "t = max(t, td)", "t = td"),
    ("vwz_generator.py", "d = marginal_mean_dwell(noise_scale) * float(-np.log(1 - u[4]))", "d = marginal_mean_dwell(noise_scale) * float(-np.log(u[4]))"),
    ("vwz_generator.py", "[3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf]", "[3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.9 * vorlauf]"),
    # vwz_models.py
    ("vwz_models.py", "np.maximum(alpha * d, (alpha - 1) * d)", "np.maximum(alpha * d, (alpha + 1) * d)"),
    ("vwz_models.py", "self.f0 = float(np.mean(y)) if self.loss == \"l2\" else float(np.quantile(y, self.alpha))", "self.f0 = float(np.median(y)) if self.loss == \"l2\" else float(np.quantile(y, self.alpha))"),
    ("vwz_models.py", "g = np.where(y > F, self.alpha, self.alpha - 1.0)", "g = np.where(y >= F, self.alpha, self.alpha - 1.0)"),
    ("vwz_models.py", "ok = (cc >= self.min_leaf) & (cnt - cc >= self.min_leaf)", "ok = (cc > self.min_leaf) & (cnt - cc > self.min_leaf)"),
    ("vwz_models.py", "if depth < self.depth and len(idx) >= 2 * self.min_leaf:", "if depth <= self.depth and len(idx) >= 2 * self.min_leaf:"),
    ("vwz_models.py", "if depth < self.depth and len(idx) >= 2 * self.min_leaf:", "if depth < self.depth and len(idx) > 2 * self.min_leaf:"),
    ("vwz_models.py", "max(1, n - p)", "max(1, n - p + 1)"),
    ("vwz_models.py", "for k in range(1, 12)]", "for k in range(1, 11)]"),
    ("vwz_models.py", "np.log(np.maximum(X[:, 6], 1e-9)) * (X[:, 6] > 0)", "np.log(np.maximum(X[:, 6], 1e-9)) * (X[:, 6] >= 0)"),
    ("vwz_models.py", "F += self.lr * self._apply(tree, Xb)\n        return self", "F += self._apply(tree, Xb)\n        return self"),
    ("vwz_models.py", "return float(np.mean(resid)) if self.loss == \"l2\" else float(np.quantile(resid, self.alpha))", "return float(np.mean(resid)) if self.loss == \"l2\" else float(np.median(resid))"),
    ("vwz_models.py", "if len(u) <= self.max_bins:", "if len(u) < self.max_bins:"),
    ("vwz_models.py", "np.exp(self.mu(X) + 0.5 * self.s ** 2)", "np.exp(self.mu(X) + self.s ** 2)"),
    # vwz_forecast.py
    ("vwz_forecast.py", "e = Est(t_arr[c] + math.exp(mu[c]))", "e = Est(t_arr[c] - math.exp(mu[c]))"),
    ("vwz_forecast.py", "key = (round(p, 2), topv if p < 0.3 else len(s), i)", "key = (round(p, 1), topv if p < 0.3 else len(s), i)"),
    ("vwz_forecast.py", "key = (round(p, 2), topv if p < 0.3 else len(s), i)", "key = (round(p, 2), topv if p < 0.5 else len(s), i)"),
    ("vwz_forecast.py", "p = float(np.dot(ws, F))", "p = 1.0 - float(np.dot(ws, F))"),
    ("vwz_forecast.py", "s = np.maximum((q90 - q10) / C.S_BAND_DIVISOR, C.S_FLOOR)", "s = np.maximum((q10 - q90) / C.S_BAND_DIVISOR, C.S_FLOOR)"),
    ("vwz_forecast.py", "return Forecast(np.exp(q50), np.exp(l2 + 0.5 * s ** 2), np.exp(q70), q50, s)", "return Forecast(np.exp(q50), np.exp(l2 - 0.5 * s ** 2), np.exp(q70), q50, s)"),
    ("vwz_forecast.py", "return Forecast(np.exp(q50), np.exp(l2 + 0.5 * s ** 2), np.exp(q70), q50, s)", "return Forecast(np.exp(q50), np.exp(l2 + 0.5 * s ** 2), np.exp(q50), q50, s)"),
    ("vwz_forecast.py", "return np.exp(ll.mu(X) + ll.s * C.Z_QUANTILE)", "return np.exp(ll.mu(X))"),
    ("vwz_forecast.py", "X, y = history(None, n_train, train_seed)", "X, y = history(0.6, n_train, train_seed)"),
    ("vwz_forecast.py", "scale = sigma * sample.mean_dwell_days", "scale = sigma"),
    ("vwz_forecast.py", "kind[1:]) / 100", "kind[1:]) / 10"),
    # vwz_evaluation.py
    ("vwz_evaluation.py", "wrong = int(np.sum(np.sign(est[a] - est[b]) != np.sign(t_dep[a] - t_dep[b])))", "wrong = int(np.sum(np.sign(est[a] - est[b]) == np.sign(t_dep[a] - t_dep[b])))"),
    ("vwz_evaluation.py", "present = (t_arr[ii] < t_dep[jj]) & (t_arr[jj] < t_dep[ii])", "present = (t_arr[ii] <= t_dep[jj]) & (t_arr[jj] <= t_dep[ii])"),
    ("vwz_evaluation.py", "present = (t_arr[ii] < t_dep[jj]) & (t_arr[jj] < t_dep[ii])", "present = (t_arr[ii] < t_dep[jj]) | (t_arr[jj] < t_dep[ii])"),
    ("vwz_evaluation.py", "return Quality(float(err.mean() / d0), wrong / tot if tot else 0.0)", "return Quality(float(err.mean()), wrong / tot if tot else 0.0)"),
    ("vwz_evaluation.py", "d0 = float(np.mean([s.mean_dwell_days for s in samples])) if d0 is None else d0", "d0 = float(np.max([s.mean_dwell_days for s in samples])) if d0 is None else d0"),
    ("vwz_evaluation.py", "return np.maximum(pred, 1e-3)", "return np.maximum(pred, 1e-2)"),
    ("vwz_evaluation.py", "est = {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.q70, C.RULE_MEAN: fc.mean, C.RULE_LOWEST_SAME: fc.q70}[key]", "est = {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.median, C.RULE_MEAN: fc.mean, C.RULE_LOWEST_SAME: fc.q70}[key]"),
    ("vwz_evaluation.py", "est = {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.q70, C.RULE_MEAN: fc.mean, C.RULE_LOWEST_SAME: fc.q70}[key]", "est = {C.RULE_MEDIAN: fc.median, C.RULE_Q70: fc.q70, C.RULE_MEAN: fc.median, C.RULE_LOWEST_SAME: fc.q70}[key]"),
    ("vwz_evaluation.py", "place = K.niedrigster_stapel if key == C.RULE_LOWEST_SAME else K.bestfit", "place = K.bestfit if key == C.RULE_LOWEST_SAME else K.bestfit"),
    ("vwz_evaluation.py", "return K.run(inst, F.true_est(sample), K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel, record=record)", "return K.run(inst, F.true_est(sample), K.niedrigster_stapel, record=record)"),
    ("vwz_evaluation.py", "return K.run(sample.inst, F.true_est(sample), K.niedrigster_stapel, reloc_rule=K.niedrigster_stapel).moves", "return K.run(sample.inst, F.true_est(sample), K.niedrigster_stapel).moves"),
    ("vwz_evaluation.py", "return K.run(inst, F.true_est(sample), K.bestfit, record=record)", "return K.run(inst, F.true_est(sample), K.niedrigster_stapel, record=record)"),
    ("vwz_evaluation.py", "return K.run(inst, F.make_est(sample.t_arr, fc.mu, fc.s), F.make_aware(), record=record)", "return K.run(inst, F.make_est(sample.t_arr, fc.mu, fc.s * 2), F.make_aware(), record=record)"),
    ("vwz_evaluation.py", "C.RULE_AWARE: fc.median, C.RULE_LOWEST_SAME: fc.q70}.get(rule_key)", "C.RULE_AWARE: fc.q70, C.RULE_LOWEST_SAME: fc.q70}.get(rule_key)"),
    ("vwz_evaluation.py", "kind = \"unclear\" if abs(diff) <= C.VERDICT_Z * se else (\"better\" if diff < 0 else \"worse\")", "kind = \"unclear\" if abs(diff) <= C.VERDICT_Z * se else (\"better\" if diff > 0 else \"worse\")"),
    ("vwz_evaluation.py", "return Verdict(kind, diff, se, 100.0 * diff / base if base else None, len(rows))", "return Verdict(kind, diff, se, 10.0 * diff / base if base else None, len(rows))"),
    ("vwz_evaluation.py", "return [r.m[key] - r.m[ref] for r in rows]", "return [r.m[ref] - r.m[key] for r in rows]"),
    ("vwz_evaluation.py", "k = max(1, -(-n // 10))", "k = max(1, n // 10)"),
    ("vwz_evaluation.py", "better = sum(1 for x in d if x < -C.STATIC_TOL) / n", "better = sum(1 for x in d if x < C.STATIC_TOL) / n"),
    ("vwz_evaluation.py", "top = sum(savings[:k]) / total if total > C.STATIC_TOL else None", "top = sum(savings[:k]) / total if total > 0.5 else None"),
    ("vwz_evaluation.py", "if d[0] >= 0:", "if d[0] > 0:"),
    ("vwz_evaluation.py", "if d[i] < 0 <= d[i + 1]:", "if d[i] <= 0 <= d[i + 1]:"),
    ("vwz_evaluation.py", "f = -d[i] / (d[i + 1] - d[i])", "f = d[i] / (d[i + 1] - d[i])"),
    ("vwz_evaluation.py", "if moves < m[0]:", "if moves <= m[0]:"),
    ("vwz_evaluation.py", "if m[i] <= moves <= m[i + 1]:", "if m[i] < moves <= m[i + 1]:"),
    ("vwz_evaluation.py", "f = (moves - m[i]) / (m[i + 1] - m[i])", "f = (moves - m[i + 1]) / (m[i + 1] - m[i])"),
    ("vwz_evaluation.py", "use = seeds if size < C.LEARNING_CURVE_BAND_BELOW else seeds[:1]", "use = seeds if size <= C.LEARNING_CURVE_BAND_BELOW else seeds[:1]"),
    ("vwz_evaluation.py", "seeds = tuple((p.train_seed + j) % (C.TRAIN_SEED_RANGE[1] + 1) for j in range(C.LEARNING_CURVE_EXTRA_SEEDS + 1))", "seeds = tuple((p.train_seed + j) for j in range(C.LEARNING_CURVE_EXTRA_SEEDS + 1))"),
    ("vwz_evaluation.py", "return Diagnosis(kind, v, p.nu is None, p.n_train < C.FEW_DATA_BELOW,", "return Diagnosis(kind, v, p.nu is None, p.n_train <= C.FEW_DATA_BELOW,"),
    ("vwz_evaluation.py", "return Diagnosis(kind, v, p.nu is None, p.n_train < C.FEW_DATA_BELOW,", "return Diagnosis(kind, v, p.nu is not None, p.n_train < C.FEW_DATA_BELOW,"),
    ("vwz_evaluation.py", "kind = {\"better\": \"helps\", \"worse\": \"tipped\", \"unclear\": \"unclear\"}[v.kind]", "kind = {\"better\": \"tipped\", \"worse\": \"helps\", \"unclear\": \"unclear\"}[v.kind]"),
    ("vwz_evaluation.py", "if occ > best_occ:", "if occ >= best_occ:"),
    ("vwz_evaluation.py", "if step.kind == \"D\" and step.moved:", "if step.kind == \"D\":"),
    ("vwz_evaluation.py", "        moves.append(float(np.mean([K.run(s.inst, e, K.bestfit).moves / nc for s, e in zip(samples, ests)])))", "        moves.append(float(np.mean([K.run(s.inst, e, K.niedrigster_stapel).moves / nc for s, e in zip(samples, ests)])))"),
    ("vwz_evaluation.py", "block = None", "block = None") if False else ("vwz_evaluation.py", "hell = float(np.mean([K.run(s.inst, F.true_est(s), K.bestfit).moves / nc for s in samples]))", "hell = float(np.mean([K.run(s.inst, F.true_est(s), K.niedrigster_stapel).moves / nc for s in samples]))"),
    ("vwz_evaluation.py", "        rows.append(Row(seed0 + i, {k: rule_moves(s, k, fc).moves / p.n_containers for k in rules}))", "        rows.append(Row(seed0 + i, {k: rule_moves(s, k, fc).moves / 300 for k in rules}))"),
    ("vwz_evaluation.py", "samples = [block_of(p, seed0 + i) for i in range(n)]", "samples = [block_of(p, seed0 + i + 1) for i in range(n)]"),
    ("vwz_evaluation.py", "return round(fill_pct", "return round(fill_pct") if False else ("vwz_evaluation.py", "return G.generate(n_stacks, max_height, fill_pct / 100, n_containers, seed, nu=nu)", "return G.generate(n_stacks, max_height, fill_pct / 10, n_containers, seed, nu=nu)"),
    # vwz_stories.py
    ("vwz_stories.py", "(d >= 0.03,", "(d > 0.03,"),
    ("vwz_stories.py", "(d <= -0.06,", "(d < -0.06,"),
    ("vwz_stories.py", "(hell <= 0.4 * rein, f\"Hellseher <= 40 % des Ausgleichs: {hell:.3f} gegen {rein:.3f}\")", "(hell < 0.4 * rein, f\"Hellseher <= 40 % des Ausgleichs: {hell:.3f} gegen {rein:.3f}\")"),
    ("vwz_stories.py", "(d >= 0.06,", "(d > 0.06,"),
    ("vwz_stories.py", "(d <= -0.18,", "(d < -0.18,"),
    ("vwz_stories.py", "(d <= -0.08,", "(d < -0.08,"),
    ("vwz_stories.py", "(rein >= 1.0,", "(rein > 1.0,"),
    ("vwz_stories.py", "(v.kind == \"worse\", f\"Urteil 'kippt' (mehr als zwei Standardfehler, SE {v.se:.3f}): {v.kind}\")", "(v.kind != \"better\", f\"Urteil 'kippt' (mehr als zwei Standardfehler, SE {v.se:.3f}): {v.kind}\")"),
    ("vwz_stories.py", "(gel >= rein + 5,", "(gel > rein + 5,"),
    ("vwz_stories.py", "(gel <= rein - 8,", "(gel < rein - 8,"),
    ("vwz_stories.py", "(hell <= 0.4 * rein, f\"Hellseher <= 40 % des Ausgleichs: {hell} gegen {rein}\")", "(hell < 0.4 * rein, f\"Hellseher <= 40 % des Ausgleichs: {hell} gegen {rein}\")"),
    ("vwz_stories.py", "(gel >= rein + 20,", "(gel > rein + 20,"),
    ("vwz_stories.py", "(gel <= 0.8 * rein,", "(gel < 0.8 * rein,"),
    ("vwz_stories.py", "(rein >= 300,", "(rein > 300,"),
    ("vwz_stories.py", "(gel <= rein - 25,", "(gel < rein - 25,"),
    ("vwz_stories.py", "return all(ok for ok, _ in block_criteria(name, moves))", "return any(ok for ok, _ in block_criteria(name, moves))"),
    # vwz_ui_panel.py
    ("vwz_ui_panel.py", "f\"{per - per_base:+.3f}\"", "f\"{per_base - per:+.3f}\""),
    ("vwz_ui_panel.py", "f\"{(base.moves - outcome.moves) / base.moves * 100:+.0f} %\"", "f\"{(outcome.moves - base.moves) / base.moves * 100:+.0f} %\""),
    ("vwz_ui_panel.py", "if is_base or not base.moves:", "if is_base:"),
    ("vwz_ui_panel.py", "if n_events < 1:", "if n_events < 0:"),
    ("vwz_ui_panel.py", "if outcome.key == C.RULE_Q70 and outcome.est_days is not None and dwell is not None:", "if outcome.key == C.RULE_Q70 and outcome.est_days is not None:"),
    # vwz_visualization.py
    ("vwz_visualization.py", "xs = [p * 100 for p in curve.pair_err]", "xs = [p * 10 for p in curve.pair_err]"),
    ("vwz_visualization.py", "rank_est = {c: r for r, c in enumerate(sorted(present, key=lambda k: (est_time[k], k)))}", "rank_est = {c: r for r, c in enumerate(sorted(present, key=lambda k: (t_dep[k], k)))}"),
    ("vwz_visualization.py", "colors.append(\"white\" if f < 0.5 else \"#1c2430\")", "colors.append(\"white\" if f <= 0.5 else \"#1c2430\")"),
    ("vwz_visualization.py", "border = C.MOVED_COLOR if c in moved else (C.ARRIVED_COLOR if c == arrived else None)", "border = C.ARRIVED_COLOR if c in moved else (C.MOVED_COLOR if c == arrived else None)"),
    ("vwz_visualization.py", "fig.update_yaxes(range=[0, top * 1.25])", "fig.update_yaxes(range=[0, top])"),
    ("vwz_visualization.py", "colors = [C.RULE_COLORS[C.RULE_Q70] if c < 0 else C.MOVED_COLOR for c in centers]", "colors = [C.RULE_COLORS[C.RULE_Q70] if c > 0 else C.MOVED_COLOR for c in centers]"),
    ("vwz_visualization.py", "fig.update_xaxes(type=\"log\", tickvals=list(lc.sizes)", "fig.update_xaxes(type=\"linear\", tickvals=list(lc.sizes)"),
    ("vwz_visualization.py", "share = counts / len(d) * 100", "share = counts / len(d)"),
    ("vwz_visualization.py", "fig.update_xaxes(fixedrange=True)", "fig.update_xaxes(fixedrange=False)"),
    ("vwz_visualization.py", "keys = [C.BASELINE] if focus_key == C.BASELINE else [C.BASELINE, focus_key]", "keys = [C.BASELINE, focus_key]"),
    # vwz_pdf_export.py
    ("vwz_pdf_export.py", "amount = f\"{abs(v.pct):.0f} % weniger\"", "amount = f\"{v.pct:.0f} % weniger\""),
    ("vwz_pdf_export.py", "\"ν\": \"nu\", ", ""),
    ("vwz_pdf_export.py", "\"σ\": \"sigma\", ", ""),
    ("vwz_pdf_export.py", "\"-\" if o.key == base.key else f\"{(o.moves - base.moves) / n:+.3f}\"", "f\"{(o.moves - base.moves) / n:+.3f}\""),
    ("vwz_pdf_export.py", "if pdf.get_y() + height > pdf.h - pdf.b_margin:", "if pdf.get_y() + height < pdf.h - pdf.b_margin:"),
    # vwz_presets.py
    ("vwz_presets.py", "value = spec.lo + round((value - spec.lo) / spec.step) * spec.step", "value = spec.lo + int((value - spec.lo) / spec.step) * spec.step"),
    ("vwz_presets.py", "        value = min(spec.hi, value)\n    if spec.step", "        value = min(spec.hi, value + 1)\n    if spec.step"),
    ("vwz_presets.py", "if spec.step and spec.step > 1 and spec.lo is not None:", "if spec.step and spec.step > 0 and spec.lo is not None:"),
    ("vwz_presets.py", "if raw not in C.LEARNERS:", "if raw not in C.LEARNERS + (C.LEARNER_GROUP,):"),
    ("vwz_presets.py", "if raw not in C.VIEW_KEYS:", "if raw not in C.RULE_KEYS:"),
    # vwz_constants.py
    ("vwz_constants.py", "VERDICT_Z = 2.0", "VERDICT_Z = 1.0"),
    ("vwz_constants.py", "VERDICT_Z = 2.0", "VERDICT_Z = 3.0"),
    ("vwz_constants.py", "FEW_DATA_BELOW = 300", "FEW_DATA_BELOW = 301"),
    ("vwz_constants.py", "SAMPLE_INSTANCES = 60", "SAMPLE_INSTANCES = 50"),
    ("vwz_constants.py", "LEARNING_CURVE_EXTRA_SEEDS = 3", "LEARNING_CURVE_EXTRA_SEEDS = 2"),
    ("vwz_constants.py", "LEARNING_CURVE_BAND_BELOW = 1000", "LEARNING_CURVE_BAND_BELOW = 3000"),
    ("vwz_constants.py", "BOOST_ROUNDS, BOOST_DEPTH, BOOST_LR, BOOST_BINS, BOOST_MIN_LEAF = 100, 3, 0.1, 64, 5", "BOOST_ROUNDS, BOOST_DEPTH, BOOST_LR, BOOST_BINS, BOOST_MIN_LEAF = 101, 3, 0.1, 64, 5"),
    ("vwz_constants.py", "BOOST_ROUNDS, BOOST_DEPTH, BOOST_LR, BOOST_BINS, BOOST_MIN_LEAF = 100, 3, 0.1, 64, 5", "BOOST_ROUNDS, BOOST_DEPTH, BOOST_LR, BOOST_BINS, BOOST_MIN_LEAF = 100, 3, 0.1, 32, 5"),
    ("vwz_constants.py", "QUANTILE = 0.7 ", "QUANTILE = 0.75 "),
    ("vwz_constants.py", "S_FLOOR = 0.05 ", "S_FLOOR = 0.10 "),
    ("vwz_constants.py", "S_BAND_DIVISOR = 2.563 ", "S_BAND_DIVISOR = 2.0 "),
    ("vwz_constants.py", "RHO = 0.7 ", "RHO = 0.8 "),
    ("vwz_constants.py", "S_BASE = (0.55, 0.25, 0.35, 0.90)", "S_BASE = (0.55, 0.25, 0.35, 0.80)"),
    ("vwz_constants.py", "ART_P = (0.35, 0.30, 0.25, 0.10)", "ART_P = (0.30, 0.35, 0.25, 0.10)"),
    ("vwz_constants.py", "GAUSS_NOISE_SEED = (13, 5) ", "GAUSS_NOISE_SEED = (13, 6) "),
    ("vwz_constants.py", "GAUSS_SIGMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)", "GAUSS_SIGMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)"),
    ("vwz_constants.py", "NU_LEVELS = (None, 1.0, 0.6, 0.3, 0.15)", "NU_LEVELS = (None, 1.0, 0.6, 0.4, 0.15)"),
    ("vwz_constants.py", "TRAIN_SIZES = (30, 100, 300, 1000, 3000, 10000)", "TRAIN_SIZES = (30, 100, 300, 1000, 3000, 5000)"),
]


# Schnelle Testdateien zuerst (ein früher Fehlschlag beendet den Lauf), die langsamen zuletzt; test_app.py und die langsamen Nachmessungen fehlen absichtlich.
TEST_FILES = ["test_stk_core.py", "test_generator.py", "test_models.py", "test_forecast.py", "test_evaluation.py", "test_stories.py", "test_presets.py", "test_visualization.py", "test_ui_panel.py",
              "test_pdf_export.py", "test_preset_stories.py"]
WORKERS = int(os.environ.get("MUT_WORKERS", "6"))         # Mutanten laufen parallel, jeder Arbeiter in einer eigenen Kopie


def make_copy():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vwz_mut_"))
    for f in ROOT.glob("*.py"):
        shutil.copy(f, tmp / f.name)
    shutil.copy(ROOT / "pytest.ini", tmp / "pytest.ini")
    shutil.copytree(ROOT / "tests", tmp / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    for f in tmp.glob("*.py"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    return tmp


def run_mutant(tmp, n, name, old, new):
    """Rückgabe: ('fehler', Anzahl) | ('gefunden' | 'zeit' | 'ueberlebt', None)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    path = tmp / name
    original = path.read_bytes().decode("utf-8")
    if original.count(old) != 1:
        return "fehler", original.count(old)
    path.write_bytes(original.replace(old, new).encode("utf-8"))
    try:
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-m", "not slow"] + [f"tests/{f}" for f in TEST_FILES], cwd=tmp, env=env, capture_output=True, text=True,
                           timeout=TIMEOUT)
        result = "ueberlebt" if r.returncode == 0 else "gefunden"
    except subprocess.TimeoutExpired:
        result = "zeit"                       # Endlosschleife: das gilt als gefunden
    finally:
        path.write_bytes(original.encode("utf-8"))
    return result, None


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    numbers = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else None          # optional: nur diese Mutanten (Nummern wie in der Ausgabe)
    todo = [(n, m) for n, m in enumerate(MUTANTS, 1) if (not only or only in m[0]) and (numbers is None or n in numbers)]
    copies = queue.Queue()
    for _ in range(max(1, min(WORKERS, len(todo)))):
        copies.put(make_copy())

    def work(item):
        n, (name, old, new) = item
        tmp = copies.get()
        try:
            res, extra = run_mutant(tmp, n, name, old, new)
        finally:
            copies.put(tmp)
        tag = {"gefunden": "gefunden", "zeit": "Zeitüberschreitung (als gefunden gezählt)", "ueberlebt": "ÜBERLEBT", "fehler": "FEHLER (Stelle nicht eindeutig)"}[res]
        print(f"[{n:3d}] {tag}  {name}: {old[:60]!r} -> {new[:60]!r}" if res in ("ueberlebt", "fehler") else f"[{n:3d}] {tag}  {name}", flush=True)
        return n, name, old, new, res, extra

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(work, todo))
    killed = sum(r[4] in ("gefunden", "zeit") for r in results)
    survivors = [r for r in results if r[4] == "ueberlebt"]
    errors = [r for r in results if r[4] == "fehler"]
    print(f"\n{killed} gefunden, {len(survivors)} überlebt, {len(errors)} Fehler in der Mutantenliste")
    for r in survivors:
        print(f"  ÜBERLEBT [{r[0]}] {r[1]}: {r[2][:70]!r} -> {r[3][:70]!r}")
    for r in errors:
        print("  FEHLER (Stelle nicht eindeutig gefunden):", r[:3], "Vorkommen:", r[5])
    while not copies.empty():
        shutil.rmtree(copies.get(), ignore_errors=True)


if __name__ == "__main__":
    main()
