"""Plotly-Figuren der Verweilzeit-Demo: Blockansicht, Paarfehler-Kurve, Lernkurve, Median/Mittelwert/Quantil, Verteilung, kumulierte Umstapelungen, Streudiagramm, Regelbalken.

Alle Achsen sind fest (fixedrange): Touch-Geräte scrollen die Seite statt im Diagramm zu zoomen oder zu schieben (Hover bleibt). Plotly wird erst in den Funktionen importiert,
damit die reine Rechnung ohne Plotly testbar bleibt."""

import numpy as np

import vwz_constants as C

# Legende unten im Container (oben rutscht sie in manchen Fenstern ins Diagramm)
LEGEND_BOTTOM = dict(orientation="h", yref="container", yanchor="bottom", y=0.0, x=0)

STACK_W = 1.0          # Breite eines Stapels (Achseneinheiten)
STACK_GAP = 0.3        # Abstand zwischen Stapeln
BOX_PAD = 0.06         # Innenabstand eines Containers im Stapel
BOX_H = 0.88           # Höhe eines Containers (Stapelebene = 1.0)


def _lock_axes(fig):
    # fixedrange: verhindert Pinch-Zoom und Ziehen, damit Touch-Geräte die Seite scrollen (Hover bleibt).
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _blend(f):
    a, b = C.BLOCK_COLOR_SOON, C.BLOCK_COLOR_LATE
    return tuple(int(round(a[i] + f * (b[i] - a[i]))) for i in range(3))


# ---------------------------------------------------------------------------------------------------
# Blockansicht
# ---------------------------------------------------------------------------------------------------
def block_title(label, moves_so_far):
    """Zweizeiliger Titel: in halbbreiten Spalten ist eine Zeile zu lang und wird abgeschnitten."""
    return f"<b>{label}</b><br><sub>Umstapelungen bisher: {moves_so_far}</sub>"


def block_figure(stacks, max_height, t_dep, est_time, title, moved=(), arrived=None, dwell=None, est_days=None):
    """Block als Stapel von Containern. Farbe = wahre Abfahrtsreihenfolge unter den anwesenden Containern, Zahl = Rang nach der Schätzung der Abfahrt (est_time, je Container).
    t_dep: wahre Abfahrtszeit je Container (Tage); dwell und est_days (optional): wahre und prognostizierte Verweilzeit für den Hover."""
    import plotly.graph_objects as go

    present = [c for s in stacks for c in s]
    rank = {c: r for r, c in enumerate(sorted(present, key=lambda k: (t_dep[k], k)))}
    rank_est = {c: r for r, c in enumerate(sorted(present, key=lambda k: (est_time[k], k)))}
    span = max(1, len(present) - 1)

    fig = go.Figure()
    n = len(stacks)
    for i in range(n):
        x0 = i * (STACK_W + STACK_GAP)
        fig.add_shape(type="rect", x0=x0, x1=x0 + STACK_W, y0=0, y1=max_height, layer="below", fillcolor=C.BLOCK_STACK_BG, line=dict(color=C.BLOCK_STACK_LINE, width=1))

    xs, ys, texts, colors, hovers = [], [], [], [], []
    for i, s in enumerate(stacks):
        x0 = i * (STACK_W + STACK_GAP)
        for t, c in enumerate(s):
            f = rank[c] / span
            r, g, b = _blend(f)
            border = C.MOVED_COLOR if c in moved else (C.ARRIVED_COLOR if c == arrived else None)
            fig.add_shape(
                type="rect", x0=x0 + BOX_PAD, x1=x0 + STACK_W - BOX_PAD, y0=t + (1 - BOX_H) / 2, y1=t + (1 - BOX_H) / 2 + BOX_H,
                fillcolor=f"rgb({r},{g},{b})", layer="below",       # unter der Text-Spur, sonst verdecken die Formen die Zahlen
                line=dict(color=border, width=3) if border else dict(width=0),
            )
            xs.append(x0 + STACK_W / 2)
            ys.append(t + 0.5)
            texts.append(str(rank_est[c] + 1))
            colors.append("white" if f < 0.5 else "#1c2430")
            state = "gerade umgestapelt" if c in moved else ("gerade angekommen" if c == arrived else "")
            hover = (f"<b>Container {c}</b>{' · ' + state if state else ''}<br>Stapel {i + 1}, Ebene {t + 1}"
                     f"<br>wahre Abfahrtsreihenfolge: {rank[c] + 1} von {len(present)}<br>nach Prognose: {rank_est[c] + 1} von {len(present)}")
            if dwell is not None and est_days is not None:
                hover += f"<br>Verweilzeit wahr {dwell[c]:.1f} d, prognostiziert {est_days[c]:.1f} d"
            hovers.append(hover)

    # eine Text-Spur trägt Zahlen UND Hover (Linien-Hover wäre punktbasiert; Zentren sind die Punkte)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=texts, textfont=dict(size=12, color=colors), hovertext=hovers, hoverinfo="text", showlegend=False))
    total_w = n * STACK_W + (n - 1) * STACK_GAP
    fig.update_layout(title=dict(text=title, font=dict(size=14), x=0.02), template="plotly_white", showlegend=False,
                      height=C.BLOCK_FIGURE_BASE_PX + 14 + max_height * C.BLOCK_FIGURE_TIER_PX, margin=dict(l=8, r=8, t=58, b=8))
    fig.update_xaxes(range=[-0.1, total_w + 0.1], visible=False)
    fig.update_yaxes(range=[-0.05, max_height + 0.05], visible=False)
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Paarfehler-Kurve (Kernabschnitt)
# ---------------------------------------------------------------------------------------------------
def pair_error_figure(curve, tipping, learned=None, current=None, learner_label=""):
    """Umstapelungen je Container über dem Anteil falsch geordneter Paare: grau das Gauß-Rauschen der Stapelplanung (8 Werte von sigma), waagerecht der Ausgleich, als Punkt der Hellseher,
    farbig die gelernten Prognosen des eingestellten Lerners (ein Punkt je Trainingsmenge, Beschriftung = Trainingscontainer), grauer Marker = die eingestellte Prognose.
    learned: Liste (Trainingsmenge, Paarfehler, Umstapelungen); current: (Paarfehler, Umstapelungen)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    xs = [p * 100 for p in curve.pair_err]
    fig.add_trace(go.Scatter(
        x=xs, y=list(curve.moves), mode="lines+markers", name="Gauß-Rauschen (Stapelplanung)", line=dict(color=C.GAUSS_COLOR, width=2.5), marker=dict(size=6, color=C.GAUSS_COLOR),
        customdata=[[s] for s in curve.sigmas], hovertemplate="<b>Gauß-Rauschen σ %{customdata[0]:.2f} Standzeiten</b><br>%{x:.0f} % falsch geordnete Paare<br>%{y:.3f} Umstapelungen je Container<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[curve.hell], mode="markers", name="Hellseher", marker=dict(size=11, color=C.RULE_COLORS[C.RULE_HELL], symbol="diamond"),
                             hovertemplate=f"<b>Hellseher</b><br>0 % falsch geordnete Paare<br>{curve.hell:.3f} Umstapelungen je Container<extra></extra>"))
    if learned:
        fig.add_trace(go.Scatter(
            x=[p * 100 for _, p, _ in learned], y=[m for _, _, m in learned], mode="lines+markers+text", name=f"gelernt: {learner_label}, Quantil 0,7",
            text=[f"{n:,}".replace(",", ".") for n, _, _ in learned], textposition="top center", textfont=dict(size=10),
            line=dict(color=C.RULE_COLORS[C.RULE_Q70], width=2), marker=dict(size=9, color=C.RULE_COLORS[C.RULE_Q70]),
            customdata=[[n] for n, _, _ in learned], hovertemplate="<b>%{customdata[0]} Trainingscontainer</b><br>%{x:.0f} % falsch geordnete Paare<br>%{y:.3f} Umstapelungen je Container<extra></extra>"))
    if current is not None:
        fig.add_trace(go.Scatter(x=[current[0] * 100], y=[current[1]], mode="markers", name="eingestellte Prognose",
                                 marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=C.MARKER_LINE_COLOR, width=3)),
                                 hovertemplate="<b>eingestellte Prognose</b><br>%{x:.0f} % falsch geordnete Paare<br>%{y:.3f} Umstapelungen je Container<extra></extra>"))
    fig.add_hline(y=curve.pure, line=dict(color=C.RULE_COLORS[C.RULE_LOWEST], width=2, dash="dash"), annotation_text="Ausgleich (ohne Prognose)", annotation_position="top left",
                  annotation_font=dict(size=11, color=C.RULE_COLORS[C.RULE_LOWEST]))
    if tipping.kind == "crosses":
        fig.add_vline(x=tipping.pair_err * 100, line=dict(color="#c0392b", width=2, dash="dash"), annotation_text=f"Kipppunkt ≈ {tipping.pair_err * 100:.0f} %", annotation_position="bottom right",
                      annotation_font=dict(size=12, color="#c0392b"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT + 60, legend=LEGEND_BOTTOM, margin=dict(t=30, b=130),
                      xaxis_title="Falsch geordnete Paare (% der gleichzeitig anwesenden Paare)", yaxis_title="Umstapelungen je Container", hovermode="closest")
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(rangemode="tozero")
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Lernkurve (Kernabschnitt)
# ---------------------------------------------------------------------------------------------------
def learning_curve_figure(lc, hell_diff, selected_learner, n_train_current):
    """Differenz zum Ausgleich (Umstapelungen je Container, negativ = besser) über der Trainingsmenge (logarithmisch): eine Linie je Lerner, für den eingestellten dazu das Band über weitere
    Trainings-Seeds; gestrichelt der Hellseher, senkrecht die eingestellte Trainingsmenge."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = {C.LEARNER_LINEAR: "#c77700", C.LEARNER_BOOSTING: C.RULE_COLORS[C.RULE_Q70], C.LEARNER_GROUP: "#8a94a3"}
    for learner in (selected_learner,) + tuple(l for l in C.LEARNERS + (C.LEARNER_GROUP,) if l != selected_learner):
        y = list(lc.series[learner])
        is_sel = learner == selected_learner
        color = colors[learner]
        if is_sel:
            lo = [b[0] for b in lc.band[learner]]
            hi = [b[1] for b in lc.band[learner]]
            fig.add_trace(go.Scatter(x=list(lc.sizes) + list(lc.sizes)[::-1], y=hi + lo[::-1], fill="toself", fillcolor=color, opacity=0.18, line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, name="Band"))
        label = C.LEARNER_LABELS[learner] + (" (Mittelwert je Art)" if learner == C.LEARNER_GROUP else ", Quantil 0,7")
        fig.add_trace(go.Scatter(x=list(lc.sizes), y=y, mode="lines+markers", name=label + (" (eingestellt)" if is_sel else ""), line=dict(color=color, width=3 if is_sel else 1.8, dash="solid" if is_sel else "dot"),
                                 marker=dict(size=7 if is_sel else 5), hovertemplate=f"<b>{C.LEARNER_LABELS[learner]}</b><br>%{{x}} Trainingscontainer<br>%{{y:+.3f}} Umstapelungen je Container gegen den Ausgleich<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.RULE_COLORS[C.RULE_LOWEST], width=2, dash="dash"), annotation_text="Ausgleich (ohne Prognose)", annotation_position="top left",
                  annotation_font=dict(size=11, color=C.RULE_COLORS[C.RULE_LOWEST]))
    fig.add_hline(y=hell_diff, line=dict(color=C.RULE_COLORS[C.RULE_HELL], width=2, dash="dot"), annotation_text="Hellseher", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=C.RULE_COLORS[C.RULE_HELL]))
    allv = [v for ser in lc.series.values() for v in ser] + [b for bands in lc.band.values() for pair in bands for b in pair] + [0.0, hell_diff]
    fig.add_trace(go.Scatter(x=[n_train_current, n_train_current], y=[min(allv), max(allv)], mode="lines", name="eingestellte Trainingsmenge", line=dict(color=C.MARKER_LINE_COLOR, width=2, dash="dot"),
                             hoverinfo="skip"))       # Spur statt add_vline: auf logarithmischen Achsen rechnen Formen und Beschriftungen unterschiedlich
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT + 60, legend=LEGEND_BOTTOM, margin=dict(t=30, b=140),
                      xaxis_title="Trainingscontainer (logarithmisch)", yaxis_title="Differenz gegen den Ausgleich (je Container)", hovermode="closest")
    fig.update_xaxes(type="log", tickvals=list(lc.sizes), ticktext=[f"{s:,}".replace(",", ".") for s in lc.sizes])
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Median oder Quantil (Kernabschnitt)
# ---------------------------------------------------------------------------------------------------
def forecast_kind_figure(rows, pure):
    """Balken: Umstapelungen je Container mit Median-, Mittelwert- und Quantil-Prognose (derselbe Lerner); unter jedem Balken der MAE: der niedrigste MAE ist nicht der niedrigste Balken.
    rows: Liste (Name, Umstapelungen je Container, MAE, Farbe)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=[f"{name}<br>MAE {mae:.2f}" for name, _, mae, _ in rows], y=[m for _, m, _, _ in rows], marker_color=[c for _, _, _, c in rows], text=[f"{m:.3f}" for _, m, _, _ in rows], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.3f} Umstapelungen je Container<extra></extra>", showlegend=False))
    fig.add_hline(y=pure, line=dict(color=C.RULE_COLORS[C.RULE_LOWEST], width=2, dash="dash"), annotation_text="Ausgleich (ohne Prognose)", annotation_position="top left",
                  annotation_font=dict(size=11, color=C.RULE_COLORS[C.RULE_LOWEST]))
    top = max([m for _, m, _, _ in rows] + [pure])
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 60, margin=dict(t=30, b=40), yaxis_title="Umstapelungen je Container")
    fig.update_yaxes(range=[0, top * 1.25])
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Verteilung der gepaarten Differenz (Kernabschnitt)
# ---------------------------------------------------------------------------------------------------
def distribution_figure(diffs, mean, label, bin_width=0.05):
    """Histogramm der gepaarten Differenz je Block (Regel minus Ausgleich, Umstapelungen je Container): grün = weniger, orange = mehr; gestrichelt das Mittel."""
    import plotly.graph_objects as go

    d = np.asarray(diffs, dtype=float)
    lo = np.floor(d.min() / bin_width) * bin_width
    hi = np.ceil(d.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    counts, edges = np.histogram(d, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    share = counts / len(d) * 100
    colors = [C.RULE_COLORS[C.RULE_Q70] if c < 0 else C.MOVED_COLOR for c in centers]
    fig = go.Figure(go.Bar(x=centers, y=share, width=bin_width * 0.92, marker_color=colors, hovertemplate="<b>%{y:.0f} % der Blöcke</b><br>Differenz um %{x:+.2f}<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line=dict(color=C.RULE_COLORS[C.RULE_LOWEST], width=2))
    fig.add_vline(x=mean, line=dict(color=C.MARKER_LINE_COLOR, width=2, dash="dash"), annotation_text=f"Mittel {mean:+.3f}", annotation_position="top", annotation_font=dict(size=11))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 100, margin=dict(t=30, b=50), xaxis_title=f"{label}: Umstapelungen je Container gegen den Ausgleich (links = besser)",
                      yaxis_title="Anteil der Blöcke (%)", bargap=0.05)
    fig.update_yaxes(rangemode="tozero")
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Kumulierte Umstapelungen, Streudiagramm, Regelbalken (Regel-Expander)
# ---------------------------------------------------------------------------------------------------
def cumulative_figure(outcomes, focus_key, cursor=None):
    """Kumulierte Umstapelungen der Regel `focus_key` gegen den Bezug über die Ereignisse."""
    import plotly.graph_objects as go

    by_key = {o.key: o for o in outcomes}
    keys = [C.BASELINE] if focus_key == C.BASELINE else [C.BASELINE, focus_key]
    fig = go.Figure()
    for key in keys:
        o = by_key[key]
        cum = (0,) + tuple(o.result.cumulative)
        fig.add_trace(go.Scatter(x=list(range(len(cum))), y=list(cum), mode="lines", name=C.RULE_SHORT[key], line=dict(color=C.RULE_COLORS[key], width=2.5, shape="hv"),
                                 hovertemplate=f"<b>{C.RULE_SHORT[key]}</b><br>nach Ereignis %{{x}}: %{{y}} Umstapelungen<extra></extra>"))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color=C.MARKER_LINE_COLOR, width=2, dash="dot"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 80, legend=LEGEND_BOTTOM, margin=dict(t=20, b=90), xaxis_title="Ereignis (Ankunft oder Abholung)",
                      yaxis_title="Umstapelungen kumuliert", hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    return _lock_axes(fig)


def scatter_figure(pred_days, dwell_days, label):
    """Prognose gegen wahre Verweilzeit (Tage, beide logarithmisch) mit Diagonale: was auf der Diagonale liegt, ist richtig prognostiziert; darüber zu lang, darunter zu kurz."""
    import plotly.graph_objects as go

    pred = np.asarray(pred_days, dtype=float)
    dwell = np.asarray(dwell_days, dtype=float)
    lo, hi = float(min(pred.min(), dwell.min())) * 0.8, float(max(pred.max(), dwell.max())) * 1.25
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=C.MARKER_LINE_COLOR, width=2, dash="dash"), name="richtig", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dwell, y=pred, mode="markers", name=label, marker=dict(size=5, color=C.RULE_COLORS[C.RULE_Q70], opacity=0.55),
                             hovertemplate="wahr %{x:.1f} d, prognostiziert %{y:.1f} d<extra></extra>"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT, legend=LEGEND_BOTTOM, margin=dict(t=20, b=90), xaxis_title="wahre Verweilzeit (Tage)", yaxis_title="Prognose (Tage)")
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    return _lock_axes(fig)


def rule_bar_figure(outcomes, n_containers):
    """Umstapelungen je Container je Regel am gezeigten Block (Balken bei 0)."""
    import plotly.graph_objects as go

    vals = [o.moves / n_containers for o in outcomes]
    fig = go.Figure(go.Bar(x=[C.RULE_SHORT[o.key] for o in outcomes], y=vals, marker_color=[C.RULE_COLORS[o.key] for o in outcomes], text=[f"{v:.3f}" for v in vals], textposition="outside",
                           hovertemplate="<b>%{x}</b><br>%{y:.3f} Umstapelungen je Container<extra></extra>", showlegend=False))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 40, margin=dict(t=30, b=60), yaxis_title="Umstapelungen je Container")
    fig.update_yaxes(range=[0, max(vals) * 1.2 + 0.01])
    return _lock_axes(fig)
