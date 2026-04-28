# ui/app.py
"""
Gradio 6.x UI — Multi-Agent AutoML System
ESI Algiers Research Demo
Run from project root: python ui/app.py
"""

from __future__ import annotations

import base64
import io
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Generator

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_datasets import get_dataset        # noqa: E402
from src.orchestrator import run_automl_pipeline  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Palette & constants
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg":       "#0b0d14",
    "s0":       "#10131f",
    "s1":       "#161929",
    "s2":       "#1e2235",
    "s3":       "#252a40",
    "border":   "#2a2f4a",
    "accent":   "#5b7cf6",
    "accent2":  "#8b6cf7",
    "green":    "#34d399",
    "red":      "#f87171",
    "yellow":   "#fbbf24",
    "text":     "#dde1f0",
    "muted":    "#6b7399",
    "idle":     "#1e2235",
}

AGENTS = [
    "DatasetQualityAgent",
    "AnalyzerAgent",
    "PlanValidatorAgent",
    "ImplementationAgent",
    "CritiqueAgent",
    "ExplainabilityAgent",
]

AGENT_SHORT = {
    "DatasetQualityAgent":  ["Dataset", "Quality"],
    "AnalyzerAgent":        ["Analyzer"],
    "PlanValidatorAgent":   ["Plan", "Validator"],
    "ImplementationAgent":  ["Implement"],
    "CritiqueAgent":        ["Critique"],
    "ExplainabilityAgent":  ["Explain-", "ability"],
}

AGENT_PREFIXES = {
    "DatasetQualityAgent":  ["DatasetQualityAgent"],
    "AnalyzerAgent":        ["AnalyzerAgent"],
    "PlanValidatorAgent":   ["PlanValidatorAgent"],
    "ImplementationAgent":  ["ImplementationAgent", "execute_plan"],
    "CritiqueAgent":        ["CritiqueAgent"],
    "ExplainabilityAgent":  ["ExplainabilityAgent"],
}

AGENT_HUE = {
    "DatasetQualityAgent":  "#fbbf24",
    "AnalyzerAgent":        "#818cf8",
    "PlanValidatorAgent":   "#34d399",
    "ImplementationAgent":  "#60a5fa",
    "CritiqueAgent":        "#f472b6",
    "ExplainabilityAgent":  "#a78bfa",
}

PRESETS = {
    "iris":     ("Iris",     "Predict the species of an iris flower from sepal and petal measurements."),
    "wine":     ("Wine",     "Predict the wine cultivar class from chemical analysis measurements."),
    "diabetes": ("Diabetes", "Predict whether a patient has diabetes based on diagnostic measurements."),
    "adult":    ("Adult",    "Predict whether an individual earns more than $50K per year from census data."),
}

# ─────────────────────────────────────────────────────────────────────────────
# Heroicons (inline SVG, 16 px)
# ─────────────────────────────────────────────────────────────────────────────

def _icon(path_d: str, size: int = 16, color: str = "currentColor") -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0">'
        f'<path d="{path_d}"/></svg>'
    )

ICON = {
    "upload":   "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12",
    "play":     "M5 3l14 9-14 9V3z",
    "check":    "M20 6L9 17l-5-5",
    "x":        "M18 6L6 18M6 6l12 12",
    "database": "M12 3C7.58 3 4 4.79 4 7s3.58 4 8 4 8-1.79 8-4-3.58-4-8-4zM4 7v5c0 2.21 3.58 4 8 4s8-1.79 8-4V7M4 12v5c0 2.21 3.58 4 8 4s8-1.79 8-4v-5",
    "cpu":      "M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2v-4M9 21H5a2 2 0 01-2-2v-4m0 0h18",
    "chart":    "M18 20V10M12 20V4M6 20v-6",
    "info":     "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
    "warning":  "M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z",
    "trophy":   "M8.21 13.89L7 23l5-3 5 3-1.21-9.12M15 7a3 3 0 11-6 0 3 3 0 016 0zM2 7h3M19 7h3M3 3l2 2M19 3l-2 2",
    "log":      "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2",
    "star":     "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
    "reset":    "M1 4v6h6M23 20v-6h-6M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15",
}

def icon_html(key: str, size: int = 15, color: str = C["muted"]) -> str:
    return _icon(ICON[key], size, color)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline SVG diagram
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_svg(
    active_agent: str | None = None,
    done_agents: list[str] | None = None,
    current_iter: int = 0,
    max_iter: int = 3,
) -> str:
    done_agents = done_agents or []
    n   = len(AGENTS)
    W   = 900
    H   = 100 if current_iter == 0 else 124
    BW, BH = 124, 48
    GAP = 12
    total = n * BW + (n - 1) * GAP
    sx  = (W - total) // 2
    cy  = 54

    g = ['<defs>',
         f'<linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0%" stop-color="#7b9fff"/><stop offset="100%" stop-color="{C["accent"]}"/></linearGradient>',
         f'<linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0%" stop-color="#6ee7a0"/><stop offset="100%" stop-color="{C["green"]}"/></linearGradient>',
         f'<linearGradient id="pg" x1="0" y1="0" x2="1" y2="0">'
         f'<stop offset="0%" stop-color="{C["accent"]}"/><stop offset="100%" stop-color="{C["accent2"]}"/></linearGradient>',
         f'<filter id="gl"><feGaussianBlur stdDeviation="2.5" result="b"/>'
         f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
         f'<marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
         f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{C["border"]}"/></marker>',
         '</defs>']

    parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;background:{C["s1"]};border-radius:10px;'
        f'border:1px solid {C["border"]}">'
    ] + g

    for i, agent in enumerate(AGENTS):
        bx = sx + i * (BW + GAP)
        by = cy - BH // 2
        ia = agent == active_agent
        id_ = agent in done_agents

        if ia:
            fill, stk, sw, fi, tc = "url(#ag)", C["accent"], 2.5, 'filter="url(#gl)"', "#fff"
        elif id_:
            fill, stk, sw, fi, tc = "url(#dg)", C["green"], 1.5, "", "#052e16"
        else:
            fill, stk, sw, fi, tc = C["idle"], C["border"], 1, "", C["muted"]

        # Arrow
        if i < n - 1:
            ax = bx + BW
            parts.append(
                f'<line x1="{ax}" y1="{cy}" x2="{ax+GAP}" y2="{cy}" '
                f'stroke="{C["border"]}" stroke-width="1.5" marker-end="url(#arr)"/>'
            )

        # Box
        parts.append(
            f'<rect x="{bx}" y="{by}" width="{BW}" height="{BH}" rx="8" '
            f'fill="{fill}" stroke="{stk}" stroke-width="{sw}" {fi}/>'
        )

        # Pulse ring
        if ia:
            parts.append(
                f'<rect x="{bx-3}" y="{by-3}" width="{BW+6}" height="{BH+6}" rx="10" '
                f'fill="none" stroke="{C["accent"]}" stroke-width="1" opacity="0.4">'
                f'<animate attributeName="opacity" values="0.4;0;0.4" dur="1.4s" repeatCount="indefinite"/></rect>'
            )

        # Label
        lines = AGENT_SHORT.get(agent, [agent])
        fw = "600" if ia else "500"
        if len(lines) == 1:
            parts.append(
                f'<text x="{bx+BW//2}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" '
                f'font-family="Inter,system-ui,sans-serif" font-size="11" font-weight="{fw}" fill="{tc}">'
                f'{lines[0]}</text>'
            )
        else:
            for li, ln in enumerate(lines):
                y0 = cy - 7 + li * 15
                parts.append(
                    f'<text x="{bx+BW//2}" y="{y0}" text-anchor="middle" dominant-baseline="middle" '
                    f'font-family="Inter,system-ui,sans-serif" font-size="11" font-weight="{fw}" fill="{tc}">'
                    f'{ln}</text>'
                )

        # Checkmark
        if id_:
            cx2, cy2 = bx + BW - 11, by + 10
            parts.append(
                f'<circle cx="{cx2}" cy="{cy2}" r="7" fill="{C["green"]}" opacity="0.95"/>'
                f'<text x="{cx2}" y="{cy2+1}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="9" font-weight="bold" fill="#052e16">✓</text>'
            )

    # Iteration bar
    if current_iter > 0:
        bar_y = H - 14
        bw = W - 48
        bx0 = 24
        pct = min(current_iter / max_iter, 1.0)
        fw2 = int(bw * pct)
        parts += [
            f'<rect x="{bx0}" y="{bar_y}" width="{bw}" height="4" rx="2" fill="{C["border"]}"/>',
            f'<rect x="{bx0}" y="{bar_y}" width="{fw2}" height="4" rx="2" fill="url(#pg)"/>',
            f'<text x="{W//2}" y="{bar_y-5}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="9.5" fill="{C["muted"]}">'
            f'Iteration {current_iter} / {max_iter}</text>',
        ]

    parts.append("</svg>")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Log panel
# ─────────────────────────────────────────────────────────────────────────────

LOG_COLORS = {
    "DatasetQualityAgent": "#fbbf24", "AnalyzerAgent": "#818cf8",
    "PlanValidatorAgent":  "#34d399",  "ImplementationAgent": "#60a5fa",
    "CritiqueAgent":       "#f472b6",  "ExplainabilityAgent": "#a78bfa",
    "Pipeline": "#94a3b8", "ITER": "#e2e8f0", "COST": "#fbbf24", "ERR": "#f87171",
}

def _lc(line: str) -> str:
    u = line.upper()
    if any(k in u for k in ("ERROR", "FAILED", "EXCEPTION")): return LOG_COLORS["ERR"]
    if "====" in line or "ITERATION" in u:                     return LOG_COLORS["ITER"]
    if "COST" in u:                                             return LOG_COLORS["COST"]
    for a, col in LOG_COLORS.items():
        if a in line:                                           return col
    if line.startswith("Pipeline"):                             return LOG_COLORS["Pipeline"]
    return C["muted"]

def logs_html(lines: list[str]) -> str:
    rows = "".join(
        f'<div style="color:{_lc(l)};margin:1px 0;white-space:pre-wrap;'
        f'word-break:break-word;line-height:1.55">'
        f'{l.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>'
        for l in lines
    )
    return (
        f'<div id="lb" style="background:{C["s1"]};border:1px solid {C["border"]};'
        f'border-radius:10px;padding:14px 16px;height:400px;overflow-y:auto;'
        f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;'
        f'box-sizing:border-box">{rows}<div id="la"></div></div>'
        f'<script>(function(){{var b=document.getElementById("lb");if(b)b.scrollTop=b.scrollHeight;}})();</script>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_agent(line: str) -> str | None:
    for agent, pfs in AGENT_PREFIXES.items():
        if any(p in line for p in pfs):
            return agent
    return None

def _aidx(a: str | None) -> int:
    try: return AGENTS.index(a) if a else -1
    except ValueError: return -1

def _detect_iter(line: str) -> int | None:
    if "==== ITERATION" in line:
        try: return int(line.split("ITERATION")[1].split("====")[0].strip())
        except: return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Results dashboard HTML
# ─────────────────────────────────────────────────────────────────────────────

def _card(body: str, p: str = "20px") -> str:
    return (
        f'<div style="background:{C["s2"]};border:1px solid {C["border"]};'
        f'border-radius:12px;padding:{p};margin-bottom:16px">{body}</div>'
    )

def _sec(label: str, icon_key: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;'
        f'padding-bottom:10px;border-bottom:1px solid {C["border"]}">'
        f'{icon_html(icon_key, 15, C["muted"])}'
        f'<span style="font-size:11px;font-weight:700;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{C["muted"]}">{label}</span></div>'
    )

def _kv(label: str, value: str, val_color: str = C["text"], mono: bool = False) -> str:
    ff = "font-family:ui-monospace,monospace;" if mono else ""
    return (
        f'<div><div style="font-size:10.5px;font-weight:600;letter-spacing:.07em;'
        f'text-transform:uppercase;color:{C["muted"]};margin-bottom:5px">{label}</div>'
        f'<div style="font-size:15px;font-weight:700;color:{val_color};{ff}">{value}</div></div>'
    )

def _fmt_steps(steps: list) -> list[str]:
    if not steps:
        return []
    parts = []
    for s in steps:
        label = s.operation.replace("_", " ").title()
        if s.method:
            label += f" · {s.method.replace('_', ' ')}"
        if s.columns:
            label += f" ({', '.join(s.columns)})"
        parts.append(label)
    return parts


def build_results_html(output: dict) -> str:
    ev   = output.get("final_evaluation") or {}
    res  = output.get("final_results")    or {}
    plan = output.get("final_plan")
    er   = output.get("explainability_report")
    qr   = output.get("quality_report")
    cr   = output.get("cost_report")

    solved     = ev.get("solved", False)
    best_model = ev.get("best_model", "—")
    best_score = ev.get("best_score", 0.0)
    suggestion = ev.get("suggestion", "")
    metric     = plan.primary_metric if plan else "score"
    iters      = output.get("iterations", "—")

    # ── Status banner ─────────────────────────────────────────────────────────
    bcolor = C["green"] if solved else C["red"]
    bicon  = icon_html("check", 16, "#fff") if solved else icon_html("x", 16, "#fff")
    banner = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:12px;background:{C["s2"]};border:1px solid {bcolor}44;'
        f'border-left:4px solid {bcolor};border-radius:12px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="width:32px;height:32px;border-radius:8px;background:{bcolor};'
        f'display:flex;align-items:center;justify-content:center">{bicon}</div>'
        f'<div><div style="font-weight:700;font-size:15px;color:{C["text"]}">'
        f'{"Problem Solved" if solved else "Not Solved"}</div>'
        f'<div style="font-size:12px;color:{C["muted"]};margin-top:1px">'
        f'Completed in {iters} iteration{"s" if iters != 1 else ""}</div></div></div>'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap">'
        f'{_kv("Best Model", best_model, C["accent"])}'
        f'{_kv("Best Score", f"{best_score:.4f}", C["accent2"], True)}'
        f'{_kv("Metric", metric)}'
        f'</div></div>'
    )

    # ── Plan details ──────────────────────────────────────────────────────────
    plan_body = ""
    if plan:
        steps = ", ".join(
            s.step_type if hasattr(s, "step_type") else str(s)
            for s in plan.preprocessing_steps
        ) if plan.preprocessing_steps else "None"
        models_tried = ", ".join(m.name for m in plan.models_to_try) if plan.models_to_try else "—"
        plan_body = _card(
            _sec("Plan Details", "cpu")
            + '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;flex-wrap:wrap">'
            + _kv("Target Column", f'<code style="background:{C["s3"]};padding:2px 8px;border-radius:5px;font-size:13px">{plan.target_column}</code>')
            + _kv("Preprocessing", steps)
            + _kv("Models Tried", models_tried)
            + '</div>'
        )
        if suggestion:
            plan_body += _card(
                f'{_sec("Suggestion for Next Run", "info")}'
                f'<p style="margin:0;color:{C["yellow"]};font-size:13.5px;line-height:1.7">{suggestion}</p>'
            )

    # ── Model comparison table ────────────────────────────────────────────────
    rows = ""
    for mname, info in res.items():
        is_best = mname == best_model
        bg = f"background:{C['s3']}" if is_best else f"background:{C['s1']}"
        td = f"padding:10px 14px;border-bottom:1px solid {C['border']}"
        if "error" in info:
            mc = f'<td style="{td};color:{C["muted"]}">—</td>'
            sc = f'<td style="{td};color:{C["muted"]}">—</td>'
            st = f'<td style="{td}"><span style="color:{C["red"]};font-size:12px;font-weight:600">Error</span></td>'
        else:
            mean = info.get("mean_score", 0)
            std  = info.get("std_score", 0)
            mc = f'<td style="{td};color:{C["text"]};font-family:monospace">{mean:.4f}</td>'
            sc = f'<td style="{td};color:{C["muted"]};font-family:monospace">±{std:.4f}</td>'
            st = f'<td style="{td}"><span style="color:{C["green"]};font-size:12px;font-weight:600">OK</span></td>'
        star = (
            f'<span style="margin-left:8px;color:{C["yellow"]}">'
            f'{icon_html("star", 12, C["yellow"])}</span>'
            if is_best else ""
        )
        nc = f'color:{C["accent"]};font-weight:700' if is_best else f'color:{C["text"]}'
        rows += (
            f'<tr style="{bg}">'
            f'<td style="{td};{nc}">{mname}{star}</td>'
            f'{mc}{sc}{st}</tr>'
        )
    th = f'padding:10px 14px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:{C["muted"]};border-bottom:2px solid {C["border"]};text-align:left'
    table_card = _card(
        f'{_sec("Model Comparison", "chart")}'
        f'<div style="overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse;font-size:13.5px">'
        f'<thead><tr><th style="{th}">Model</th><th style="{th}">Mean Score</th>'
        f'<th style="{th}">Std Dev</th><th style="{th}">Status</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )

    # ── Feature importance ────────────────────────────────────────────────────
    fi_card = ""
    if er and er.top_features:
        top5  = er.top_features[:5]
        mx    = max((f.importance_score for f in top5), default=1) or 1
        bcolors = [C["accent"], C["accent2"], "#60a5fa", C["green"], "#f472b6"]
        bars  = ""
        for i, feat in enumerate(top5):
            pct = (feat.importance_score / mx) * 100
            bc  = bcolors[i % len(bcolors)]
            bars += (
                f'<div style="margin-bottom:14px">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">'
                f'<span style="color:{C["text"]};font-size:13px">'
                f'<span style="color:{C["muted"]};margin-right:8px;font-size:11px">{feat.rank:02d}</span>'
                f'{feat.feature_name}</span>'
                f'<span style="color:{bc};font-weight:700;font-family:monospace;font-size:13px">{feat.importance_score:.4f}</span>'
                f'</div>'
                f'<div style="background:{C["border"]};border-radius:3px;height:6px">'
                f'<div style="width:{pct:.1f}%;background:{bc};border-radius:3px;height:6px"></div>'
                f'</div></div>'
            )
        method_pill = (
            f'<span style="background:{C["s3"]};border:1px solid {C["border"]};'
            f'border-radius:6px;padding:2px 10px;font-size:11px;color:{C["muted"]};'
            f'margin-left:8px;font-family:monospace">{er.method}</span>'
        )
        fi_card = _card(
            f'{_sec("Feature Importances", "chart")}'
            f'<div style="margin-bottom:16px">{method_pill}</div>'
            f'{bars}'
        )

    # ── Decision summary ──────────────────────────────────────────────────────
    ds_card = ""
    if er and er.decision_summary:
        ds_card = _card(
            f'{_sec("Model Decision Summary", "info")}'
            f'<p style="margin:0;color:{C["text"]};font-size:13.5px;line-height:1.8">'
            f'{er.decision_summary}</p>'
        )

    # ── Quality warnings ──────────────────────────────────────────────────────
    qw_card = ""
    if qr:
        all_w = list(getattr(qr, "warnings", []) or []) + list(getattr(qr, "issues", []) or [])
        if all_w:
            items = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:10px;'
                f'padding:10px 0;border-bottom:1px solid {C["border"]}">'
                f'{icon_html("warning", 14, C["yellow"])}'
                f'<span style="color:{C["text"]};font-size:13px">{w}</span></div>'
                for w in all_w
            )
            qw_card = _card(f'{_sec("Dataset Quality Warnings", "warning")}{items}')

    # ── Cost summary ──────────────────────────────────────────────────────────
    cost_card = ""
    if cr:
        try:
            summary = cr.summary()
            cost_card = _card(
                f'{_sec("Cost Report", "info")}'
                f'<pre style="margin:0;color:{C["muted"]};font-size:12px;'
                f'white-space:pre-wrap;font-family:monospace;line-height:1.7">{summary}</pre>'
            )
        except Exception:
            pass

    return (
        f'<div style="font-family:Inter,ui-sans-serif,sans-serif;color:{C["text"]};padding:2px">'
        f'{banner}{plan_body}{table_card}{fi_card}{ds_card}{qw_card}{cost_card}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline thread
# ─────────────────────────────────────────────────────────────────────────────

def _run_thread(csv_path: str, problem: str, holder: dict, q: queue.Queue):
    try:
        output = run_automl_pipeline(
            csv_path=csv_path,
            problem=problem,
            max_iterations=3,
            use_explainability=True,
            log_callback=lambda msg: q.put(("log", msg)),
        )
        holder["output"] = output
    except Exception:
        q.put(("error", traceback.format_exc()))
    finally:
        q.put(("done", None))


def run_pipeline(csv_file, problem_text: str, preset_csv: str, preset_problem: str) -> Generator:
    # Resolve inputs
    if csv_file is not None:
        csv_path = csv_file if isinstance(csv_file, str) else csv_file.name
        problem  = problem_text.strip()
    elif preset_csv:
        csv_path = preset_csv
        problem  = preset_problem or problem_text.strip()
    else:
        yield (build_pipeline_svg(), logs_html(["No dataset selected."]), "", "**Status:** Idle")
        return
    if not problem:
        yield (build_pipeline_svg(), logs_html(["No problem statement."]), "", "**Status:** Idle")
        return

    q: queue.Queue = queue.Queue()
    holder: dict   = {}
    threading.Thread(target=_run_thread, args=(csv_path, problem, holder, q), daemon=True).start()

    active: str | None = "DatasetQualityAgent"
    done:   list[str]  = []
    shown:  list[str]  = ["Pipeline started."]
    cur_it: int        = 0

    yield (build_pipeline_svg(active, done, cur_it), logs_html(shown), "", "**Status:** Running…")

    while True:
        try:
            kind, val = q.get(timeout=0.25)
        except queue.Empty:
            yield (build_pipeline_svg(active, done, cur_it), logs_html(shown + ["…"]), "", "**Status:** Running…")
            continue

        if kind == "log":
            shown.append(val)
            it = _detect_iter(val)
            if it:
                cur_it = it
                done   = []
                active = "DatasetQualityAgent"
            det = _detect_agent(val)
            if det and _aidx(det) > _aidx(active):
                if active and active not in done:
                    done.append(active)
                active = det
            yield (build_pipeline_svg(active, done, cur_it), logs_html(shown), "", "**Status:** Running…")

        elif kind == "error":
            err_html = (
                f'<p style="color:{C["red"]};padding:20px;font-family:sans-serif">'
                f'Pipeline error — see logs tab.</p>'
            )
            yield (
                build_pipeline_svg(),
                logs_html(shown + [""] + val.splitlines()),
                err_html,
                "**Status:** Error",
            )
            return

        elif kind == "done":
            break

    results_html = build_results_html(holder.get("output", {}))
    yield (
        build_pipeline_svg(None, list(AGENTS), cur_it, 3),
        logs_html(shown),
        results_html,
        "**Status:** Complete",
    )


# ─────────────────────────────────────────────────────────────────────────────
# QR code generator (for deployment)
# ─────────────────────────────────────────────────────────────────────────────

def make_qr_html(url: str) -> str:
    try:
        import qrcode, qrcode.image.svg
        img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
        buf = io.BytesIO()
        img.save(buf)
        svg_bytes = buf.getvalue().decode()
        # Extract just the <svg> tag
        svg_start = svg_bytes.find("<svg")
        svg = svg_bytes[svg_start:]
        # Style it
        svg = svg.replace("<svg ", f'<svg style="width:100%;max-width:180px;height:auto;" ', 1)
        return (
            f'<div style="text-align:center;padding:16px">'
            f'<div style="display:inline-block;background:#fff;padding:12px;border-radius:12px">{svg}</div>'
            f'<div style="margin-top:10px;font-size:11px;color:{C["muted"]};font-family:monospace">{url}</div>'
            f'</div>'
        )
    except Exception as e:
        return f'<div style="color:{C["muted"]};font-size:12px;padding:16px">QR: {e}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; }}

body, .gradio-container, .gradio-container > .main {{
    background: {C["bg"]} !important;
    color: {C["text"]} !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
    min-height: 100vh;
}}

/* Wipe Gradio's default panel chrome */
.block, .form, .gap, .panel, .gr-box {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

/* Tabs */
.tab-nav {{
    background: {C["s1"]} !important;
    border-bottom: 1px solid {C["border"]} !important;
    border-radius: 0 !important;
    padding: 0 8px !important;
}}
.tab-nav button {{
    color: {C["muted"]} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color .15s, border-color .15s !important;
}}
.tab-nav button.selected {{
    color: {C["accent"]} !important;
    border-bottom-color: {C["accent"]} !important;
    font-weight: 600 !important;
}}
.tabitem {{ padding: 20px !important; background: {C["bg"]} !important; }}

/* File upload */
.upload-container, [data-testid="file-upload"] {{
    background: {C["s2"]} !important;
    border: 1px dashed {C["border"]} !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}}
.upload-container:hover {{ border-color: {C["accent"]} !important; }}

/* Textarea */
textarea, input[type=text] {{
    background: {C["s2"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    padding: 10px 12px !important;
    transition: border-color .2s !important;
    resize: vertical !important;
}}
textarea:focus, input[type=text]:focus {{
    border-color: {C["accent"]} !important;
    outline: none !important;
}}
label span {{ color: {C["muted"]} !important; font-size: 11.5px !important; font-weight: 600 !important; letter-spacing: .05em !important; text-transform: uppercase !important; }}

/* Preset buttons */
.preset-btn button {{
    background: {C["s2"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    height: 40px !important;
    transition: border-color .2s, background .2s !important;
    width: 100% !important;
}}
.preset-btn button:hover {{
    border-color: {C["accent"]} !important;
    color: {C["accent"]} !important;
    background: {C["s1"]} !important;
}}

/* Run button */
#run_btn button, #run_btn {{
    background: linear-gradient(135deg, {C["accent"]}, {C["accent2"]}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: 44px !important;
    letter-spacing: .03em !important;
    transition: opacity .2s, transform .1s !important;
    width: 100% !important;
    cursor: pointer !important;
}}
#run_btn button:hover {{ opacity: .88 !important; }}
#run_btn button:active {{ transform: scale(.98) !important; }}

/* Status */
#status_row {{ padding: 0 !important; }}
#status_md p {{ margin: 0 !important; font-size: 12.5px !important; color: {C["muted"]} !important; }}

/* Markdown in results */
.results-tab p {{ color: {C["text"]}; line-height: 1.7; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C["border"]}; border-radius: 4px; }}

/* Remove Gradio footer */
footer {{ display: none !important; }}
.built-with {{ display: none !important; }}

/* Dividers */
.divider {{
    height: 1px;
    background: {C["border"]};
    margin: 16px 0;
}}

/* Section label */
.sec-label {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: {C["muted"]};
    margin-bottom: 10px;
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Header HTML
# ─────────────────────────────────────────────────────────────────────────────

HEADER_HTML = (
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'
    f'padding:18px 24px;background:{C["s1"]};border-bottom:1px solid {C["border"]}">'
    f'<div style="display:flex;align-items:center;gap:14px">'
    f'<div style="width:40px;height:40px;border-radius:10px;flex-shrink:0;'
    f'background:linear-gradient(135deg,{C["accent"]},{C["accent2"]});'
    f'display:flex;align-items:center;justify-content:center">'
    + _icon(ICON["cpu"], 20, "#fff") +
    f'</div>'
    f'<div>'
    f'<div style="font-size:18px;font-weight:800;color:{C["text"]};letter-spacing:-.02em;line-height:1.2">'
    f'Multi-Agent AutoML</div>'
    f'<div style="font-size:12px;color:{C["muted"]};margin-top:2px">'
    f'ESI Algiers &nbsp;&middot;&nbsp; Research Demo &nbsp;&middot;&nbsp;'
    f'<span style="color:{C["accent"]};font-weight:600">6 LLM Agents</span></div>'
    f'</div></div>'
    f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
    # + "".join(
    #     f'<span style="background:{C["s3"]};border:1px solid {C["border"]};'
    #     f'border-radius:6px;padding:3px 10px;font-size:11px;color:{C["muted"]}">{a}</span>'
    #     for a in ["DatasetQualityAgent", "AnalyzerAgent", "PlanValidatorAgent",
    #               "ImplementationAgent", "CritiqueAgent", "ExplainabilityAgent"]
    # ) 
    +
    f'</div></div>'
)

def _sec_label(label: str, icon_key: str) -> str:
    return (
        f'<div class="sec-label">'
        f'{icon_html(icon_key, 13, C["muted"])}'
        f'<span>{label}</span></div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Multi-Agent AutoML") as demo:
        gr.HTML(HEADER_HTML)

        hidden_csv     = gr.Textbox(value="", visible=False)
        hidden_problem = gr.Textbox(value="", visible=False)

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — Run
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("  Run Pipeline  "):
                with gr.Row(equal_height=False):

                    # ── Left panel: inputs ────────────────────────────────────
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML(
                            f'<div style="background:{C["s1"]};border:1px solid {C["border"]};'
                            f'border-radius:12px;padding:18px;height:100%">'
                            f'{_sec_label("Dataset", "database")}'
                        )
                        csv_upload = gr.File(label="Upload CSV", file_types=[".csv"], height=80)
                        gr.HTML(
                            f'<div style="text-align:center;color:{C["muted"]};'
                            f'font-size:11px;margin:10px 0">— or select a preset —</div>'
                        )
                        with gr.Row():
                            btn_iris  = gr.Button("Iris",     elem_classes=["preset-btn"], size="sm")
                            btn_wine  = gr.Button("Wine",     elem_classes=["preset-btn"], size="sm")
                        with gr.Row():
                            btn_diab  = gr.Button("Diabetes", elem_classes=["preset-btn"], size="sm")
                            btn_adult = gr.Button("Adult",    elem_classes=["preset-btn"], size="sm")

                        preset_info = gr.HTML(
                            f'<div style="min-height:18px;font-size:11.5px;color:{C["accent"]};'
                            f'text-align:center;margin-top:4px"></div>'
                        )

                        gr.HTML(f'<div class="divider"></div>{_sec_label("Problem Statement", "log")}')

                        problem_input = gr.Textbox(
                            label="",
                            placeholder="Describe the prediction task, e.g. Predict whether a patient has diabetes…",
                            lines=5,
                            max_lines=10,
                        )

                        gr.HTML('<div style="margin-top:12px"></div>')
                        run_btn   = gr.Button("Run Pipeline", elem_id="run_btn", variant="primary")
                        gr.HTML('<div style="margin-top:8px"></div>')
                        status_md = gr.Markdown("**Status:** Idle", elem_id="status_md")
                        gr.HTML('</div>')  # close card div

                    # ── Right panel: pipeline diagram + logs ──────────────────
                    with gr.Column(scale=2, min_width=480):
                        gr.HTML(
                            f'<div style="background:{C["s1"]};border:1px solid {C["border"]};'
                            f'border-radius:12px;padding:18px">'
                            f'{_sec_label("Agent Pipeline", "cpu")}'
                        )
                        pipeline_svg = gr.HTML(build_pipeline_svg())

                        gr.HTML(
                            f'<div class="divider"></div>'
                            f'{_sec_label("Live Execution Log", "log")}'
                        )
                        log_panel = gr.HTML(logs_html(["Waiting to start…"]))
                        gr.HTML('</div>')

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — Results
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("  Results  "):
                results_panel = gr.HTML(
                    f'<div style="padding:40px;text-align:center;color:{C["muted"]};font-size:14px;'
                    f'border:1px dashed {C["border"]};border-radius:12px;margin:8px 0">'
                    f'{icon_html("chart", 32, C["border"])}'
                    f'<p style="margin-top:12px">Run the pipeline to see results here.</p></div>'
                )

        # ── Preset callbacks ──────────────────────────────────────────────────
        def _preset(slug: str):
            info = get_dataset(slug)
            label, _ = PRESETS[slug]
            pill = (
                f'<div style="font-size:11.5px;color:{C["accent"]};text-align:center;margin-top:4px">'
                f'{icon_html("database", 12, C["accent"])} '
                f'{label} &nbsp;·&nbsp; {info.n_rows} rows, {info.n_cols} cols</div>'
            )
            return info.csv_path, info.problem_statement, info.problem_statement, pill

        for btn, slug in [
            (btn_iris, "iris"), (btn_wine, "wine"),
            (btn_diab, "diabetes"), (btn_adult, "adult"),
        ]:
            btn.click(
                fn=lambda s=slug: _preset(s),
                outputs=[hidden_csv, hidden_problem, problem_input, preset_info],
            )

        csv_upload.change(
            fn=lambda f: ("", "", f'<div style="min-height:18px"></div>'),
            inputs=[csv_upload],
            outputs=[hidden_csv, hidden_problem, preset_info],
        )

        # ── Run ───────────────────────────────────────────────────────────────
        run_btn.click(
            fn=run_pipeline,
            inputs=[csv_upload, problem_input, hidden_csv, hidden_problem],
            outputs=[pipeline_svg, log_panel, results_panel, status_md],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = build_app()

    if args.share:
        print("\n  Generating QR code for the public link…")

    _, local_url, share_url = app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        quiet=True,
    )

    url = share_url or local_url or f"http://{args.host}:{args.port}"
    print(f"\n  App running at: {url}")

    if args.share and share_url:
        try:
            import qrcode, qrcode.image.svg
            qr = qrcode.QRCode(border=2)
            qr.add_data(share_url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
            print(f"\n  Share URL: {share_url}\n")
        except ImportError:
            print(f"  Install qrcode for QR: pip install 'qrcode[pil]'\n")

    app.block()