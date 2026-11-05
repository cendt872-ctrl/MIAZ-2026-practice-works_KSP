#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практична робота з дисципліни:
«Методологія відображення даних в інтересах ІАЗ»
Заняття 3: «Практичне використання інструментів візуалізації даних»
Повна реалізація всіх 10 варіантів завдань.

УВАГА: Усі дані в цьому скрипті є СИНТЕТИЧНИМИ НАВЧАЛЬНИМИ ДАНИМИ,
згенерованими з фіксованим початковим значенням генератора випадкових чисел (seed=42)
виключно для демонстрації методів очищення, аналізу та візуалізації даних.
Вони НЕ відображають фактичних відомостей про реальні бойові дії чи реальні підрозділи.
"""

import os
import sys
import json
import math
import base64
import webbrowser
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats


# ==============================================================================
# 0. НАЛАШТУВАННЯ СЕРЕДОВИЩА, ПАЛІТРИ ТА ШРИФТІВ
# ==============================================================================
SEED = 42
np.random.seed(SEED)

# Стримана тематична палітра (olive, earth tones, NATO blue) — умовна аналітична
# палітра високої контрастності (не є офіційним військовим стандартом)
PALETTE = {
    "nato_blue": "#1F4E78",        # Глибокий синій (NATO blue)
    "olive_dark": "#4B5320",       # Темно-оливковий
    "olive_mid": "#6B7B3A",        # Середньо-оливковий
    "earth_ochre": "#B87333",      # Земляний вохристий / мідний
    "earth_terracotta": "#A04030", # Теракотовий (акцент високої інтенсивності)
    "earth_sand": "#C2B280",       # Пісочний
    "slate_grey": "#4A5568",       # Сланцево-сірий
    "bg_light": "#F7F6F2",         # Світлий нейтральний фон
    "grid_grey": "#D8D6CE",        # Сітка
    "text_dark": "#1C2321"         # Основний текст
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"],
    "axes.edgecolor": "#4A5568",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": PALETTE["grid_grey"],
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAF9F5",
    "text.color": PALETTE["text_dark"],
    "axes.labelcolor": PALETTE["text_dark"],
    "xtick.color": PALETTE["text_dark"],
    "ytick.color": PALETTE["text_dark"],
    "figure.dpi": 160,
    "savefig.dpi": 160,
})


def setup_dirs(base_out_dir: str):
    """Створює структуру каталогів для збереження результатів."""
    csv_dir = os.path.join(base_out_dir, "data_csv")
    png_dir = os.path.join(base_out_dir, "figures_png")
    html_dir = os.path.join(base_out_dir, "interactive_html")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    return csv_dir, png_dir, html_dir


# ==============================================================================
# ДОПОМІЖНІ АЛГОРИТМИ (ДЛЯ ПОВНОЇ ВІДТВОРЮВАНОСТІ БЕЗ ЗОВНІШНІХ ЗАЛЕЖНОСТЕЙ)
# ==============================================================================
def squarify_rects(sizes, x, y, dx, dy):
    """
    Обчислює прямокутники для побудови ієрархічної діаграми Treemap
    (реалізація алгоритму Bruls-Huizing-van Wijk squarified treemap).
    """
    sizes = np.array(sizes, dtype=float)
    sizes = sizes / sizes.sum() * dx * dy

    def layout_row(row, rx, ry, rdx, rdy):
        area = sum(row)
        rects = []
        if rdx >= rdy:
            w = area / rdy if rdy > 0 else 0
            cy = ry
            for s in row:
                h = s / w if w > 0 else 0
                rects.append((rx, cy, w, h))
                cy += h
        else:
            h = area / rdx if rdx > 0 else 0
            cx = rx
            for s in row:
                w = s / h if h > 0 else 0
                rects.append((cx, ry, w, h))
                cx += w
        return rects

    def worst_ratio(row, w):
        if not row or w <= 0:
            return float("inf")
        s = sum(row)
        r_max = max(row)
        r_min = min(row)
        return max((w**2 * r_max) / (s**2), (s**2) / (w**2 * r_min))

    rects = []
    remaining = list(sizes)
    cx, cy, cdx, cdy = x, y, dx, dy
    while remaining:
        w = min(cdx, cdy)
        row = [remaining[0]]
        remaining = remaining[1:]
        while remaining and worst_ratio(row + [remaining[0]], w) <= worst_ratio(row, w):
            row.append(remaining[0])
            remaining = remaining[1:]
        rects.extend(layout_row(row, cx, cy, cdx, cdy))
        area = sum(row)
        if cdx >= cdy:
            dw = area / cdy if cdy > 0 else 0
            cx += dw
            cdx -= dw
        else:
            dh = area / cdx if cdx > 0 else 0
            cy += dh
            cdy -= dh
    return rects


def compute_directed_centralities(nodes, edges):
    """
    Точний розрахунок показників центральності для орієнтованого графа G=(V,E):
    - In-Degree Centrality: k_in(v) / (N - 1)
    - Out-Degree Centrality: k_out(v) / (N - 1)
    - Total Degree Centrality: (k_in(v) + k_out(v)) / (N - 1)
    - Closeness Centrality (Wasserman-Faust outgoing shortest paths)
    - Betweenness Centrality (точний алгоритм Брандеса для орієнтованих графів,
      нормалізація 1 / ((N - 1)*(N - 2)))
    """
    n = len(nodes)
    adj_out = {u: [] for u in nodes}
    adj_in = {u: [] for u in nodes}
    for u, v, *_ in edges:
        if v not in adj_out[u]:
            adj_out[u].append(v)
        if u not in adj_in[v]:
            adj_in[v].append(u)

    in_deg = {u: len(adj_in[u]) / (n - 1) for u in nodes}
    out_deg = {u: len(adj_out[u]) / (n - 1) for u in nodes}
    tot_deg = {u: (len(adj_in[u]) + len(adj_out[u])) / (n - 1) for u in nodes}

    closeness = {}
    for s in nodes:
        dist = {s: 0}
        q = deque([s])
        while q:
            curr = q.popleft()
            for nxt in adj_out[curr]:
                if nxt not in dist:
                    dist[nxt] = dist[curr] + 1
                    q.append(nxt)
        reachable = len(dist) - 1
        total_dist = sum(dist.values())
        if reachable > 0 and total_dist > 0:
            closeness[s] = (reachable / total_dist) * (reachable / (n - 1))
        else:
            closeness[s] = 0.0

    betweenness = {u: 0.0 for u in nodes}
    for s in nodes:
        stack = []
        pred = {w: [] for w in nodes}
        sigma = {w: 0.0 for w in nodes}
        sigma[s] = 1.0
        dist = {w: -1 for w in nodes}
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj_out[v]:
                if dist[w] < 0:
                    q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {w: 0.0 for w in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    norm = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    for u in nodes:
        betweenness[u] *= norm

    return pd.DataFrame({
        "node_id": nodes,
        "in_degree_centrality": [round(in_deg[u], 4) for u in nodes],
        "out_degree_centrality": [round(out_deg[u], 4) for u in nodes],
        "total_degree_centrality": [round(tot_deg[u], 4) for u in nodes],
        "betweenness_centrality": [round(betweenness[u], 4) for u in nodes],
        "closeness_centrality": [round(closeness[u], 4) for u in nodes],
        "in_links_count": [len(adj_in[u]) for u in nodes],
        "out_links_count": [len(adj_out[u]) for u in nodes],
    })


# ==============================================================================
# ВАРІАНТ 1: КАРТА ПОДІЙ (ГЕОПРОСТОРОВЕ ВІДОБРАЖЕННЯ ТА ФІЛЬТРУВАННЯ ЗА ЧАСОМ)
# ==============================================================================
def run_variant_1(csv_dir, png_dir, html_dir):
    rng = np.random.default_rng(SEED + 1)
    sectors = [
        ("Північний рубіж", 49.65, 32.15),
        ("Східний виступ", 49.45, 32.65),
        ("Південна ділянка", 49.18, 32.35),
        ("Західний вузол", 49.42, 31.85),
    ]
    event_types = [
        "Артилерійський обстріл",
        "Активність БПЛА",
        "Штурмова дія",
        "Радіоелектронне подавлення",
    ]
    dates = pd.date_range("2026-10-01", "2026-10-14", freq="D")

    records = []
    ev_id = 1001
    for d in dates:
        n_ev = rng.integers(8, 15)
        for _ in range(n_ev):
            sec_name, base_lat, base_lon = sectors[rng.integers(0, len(sectors))]
            ev_type = rng.choice(event_types, p=[0.40, 0.30, 0.18, 0.12])
            lat = round(base_lat + rng.normal(0, 0.07), 5)
            lon = round(base_lon + rng.normal(0, 0.09), 5)
            intensity = round(float(rng.uniform(2.0, 9.8)), 1)
            duration_min = int(rng.integers(10, 145))
            records.append({
                "event_id": f"EV-{ev_id}",
                "timestamp": f"{d.strftime('%Y-%m-%d')} {rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00",
                "date": d.strftime("%Y-%m-%d"),
                "sector": sec_name,
                "event_type": ev_type,
                "latitude": lat,
                "longitude": lon,
                "intensity_index": intensity,
                "duration_min": duration_min,
                "is_synthetic_training_data": True
            })
            ev_id += 1

    df_raw = pd.DataFrame(records)
    dups = df_raw.iloc[[2, 15, 40]].copy()
    swapped = df_raw.iloc[[5, 22, 55, 88]].copy()
    for idx in swapped.index:
        df_raw.loc[idx, "latitude"], df_raw.loc[idx, "longitude"] = (
            df_raw.loc[idx, "longitude"],
            df_raw.loc[idx, "latitude"],
        )
    df_raw.loc[[10, 33, 71], "intensity_index"] = np.nan
    df_raw.loc[[19, 64], "intensity_index"] = -3.5

    df_raw = pd.concat([df_raw, dups], ignore_index=True)
    raw_path = os.path.join(csv_dir, "v1_events_raw.csv")
    df_raw.to_csv(raw_path, index=False, encoding="utf-8")

    n_initial = len(df_raw)
    df_clean = df_raw.drop_duplicates(subset=["event_id"]).copy()
    n_dups_removed = n_initial - len(df_clean)

    swap_mask = (df_clean["latitude"] < 40.0) & (df_clean["longitude"] > 44.0)
    n_swapped_fixed = int(swap_mask.sum())
    df_clean.loc[swap_mask, ["latitude", "longitude"]] = df_clean.loc[
        swap_mask, ["longitude", "latitude"]
    ].values

    invalid_mask = df_clean["intensity_index"].isna() | (df_clean["intensity_index"] <= 0)
    n_invalid_removed = int(invalid_mask.sum())
    df_clean = df_clean[~invalid_mask].reset_index(drop=True)

    clean_path = os.path.join(csv_dir, "v1_events_clean.csv")
    df_clean.to_csv(clean_path, index=False, encoding="utf-8")

    color_map = {
        "Артилерійський обстріл": PALETTE["earth_terracotta"],
        "Активність БПЛА": PALETTE["nato_blue"],
        "Штурмова дія": PALETTE["earth_ochre"],
        "Радіоелектронне подавлення": PALETTE["olive_dark"],
    }

    fig, (ax_map, ax_time) = plt.subplots(
        1, 2, figsize=(13.8, 6.0), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    for ev_type, grp in df_clean.groupby("event_type"):
        ax_map.scatter(
            grp["longitude"],
            grp["latitude"],
            s=grp["intensity_index"] * 18,
            c=color_map[ev_type],
            alpha=0.78,
            edgecolors="white",
            linewidth=0.7,
            label=f"{ev_type} (n={len(grp)})",
        )
    for sec_name, blat, blon in sectors:
        ax_map.annotate(
            sec_name,
            (blon, blat),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["slate_grey"], alpha=0.85),
        )
    ax_map.set_title("Просторова локалізація умовних подій (WGS-84, EPSG:4326)", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("Довгота (Longitude, ° сх. д.)")
    ax_map.set_ylabel("Широта (Latitude, ° пн. ш.)")
    ax_map.legend(loc="lower left", fontsize=8, title="Тип події (розмір = інтенсивність, бали)")

    daily_type = df_clean.groupby(["date", "event_type"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(daily_type))
    x_idx = np.arange(len(daily_type))
    for ev_type in event_types:
        vals = daily_type[ev_type].values if ev_type in daily_type else np.zeros(len(daily_type))
        ax_time.bar(x_idx, vals, bottom=bottom, color=color_map[ev_type], label=ev_type, width=0.72, alpha=0.9)
        bottom += vals
    ax_time.set_xticks(x_idx[::2])
    ax_time.set_xticklabels([d[5:] for d in daily_type.index[::2]], rotation=30)
    ax_time.set_title("Добовий розподіл кількості подій за типами (01–14.10.2026)", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Дата спостереження (ММ-ДД)")
    ax_time.set_ylabel("Кількість зафіксованих подій, од./добу")
    ax_time.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Варіант 1. Карта умовних подій та їхня інтенсивність (Синтетичні навчальні дані, seed=42)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = os.path.join(png_dir, "v1_events_map.png")
    fig.savefig(png_path)
    plt.close(fig)

    html_path = os.path.join(html_dir, "v1_events_interactive_map.html")
    events_json = df_clean.to_dict(orient="records")
    html_content = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Варіант 1 — Інтерактивна карта подій (Синтетичні навчальні дані)</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #F7F6F2; color: #1C2321; }}
  h1 {{ font-size: 18px; color: #1F4E78; margin: 0 0 6px 0; }}
  .subtitle {{ font-size: 13px; color: #4A5568; margin-bottom: 14px; }}
  .controls {{ display: flex; gap: 18px; align-items: center; background: #fff; padding: 12px 16px; border: 1px solid #D8D6CE; margin-bottom: 14px; flex-wrap: wrap; }}
  .map-box {{ display: flex; gap: 16px; }}
  svg {{ background: #EEF2EB; border: 1px solid #4A5568; flex: 2; height: 520px; }}
  .side-panel {{ flex: 1; background: #fff; border: 1px solid #D8D6CE; padding: 14px; height: 492px; overflow-y: auto; }}
  .tooltip {{ position: fixed; pointer-events: none; background: rgba(28,35,33,0.94); color: #fff; padding: 8px 10px; font-size: 12px; border-radius: 4px; display: none; z-index: 1000; }}
  .badge {{ display: inline-block; width: 11px; height: 11px; margin-right: 6px; border-radius: 50%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #E2E8F0; padding: 5px 6px; text-align: left; }}
  th {{ color: #1F4E78; border-bottom: 2px solid #1F4E78; }}
</style>
</head>
<body>
<h1>Варіант 1. Інтерактивна карта умовних подій (WGS-84, EPSG:4326) — Навчальний приклад</h1>
<div class="subtitle">Кольори позначають тип події, радіус маркера — інтенсивність (за 10-бальною шкалою). Використовуйте фільтри дати та типу події.</div>
<div class="controls">
  <label><b>Фільтр за датою:</b>
    <select id="dateFilter"><option value="ALL">Усі дати (01.10–14.10.2026)</option></select>
  </label>
  <label><b>Тип події:</b>
    <select id="typeFilter"><option value="ALL">Усі типи подій</option></select>
  </label>
  <label><b>Мін. інтенсивність (бали):</b>
    <input type="range" id="intFilter" min="1" max="9" step="0.5" value="1">
    <span id="intVal">1.0</span>
  </label>
  <span id="counter" style="margin-left:auto;font-weight:bold;color:#1F4E78;"></span>
</div>
<div class="map-box">
  <svg id="mapSvg" viewBox="0 0 760 520"></svg>
  <div class="side-panel">
    <h3 style="margin-top:0;font-size:14px;color:#1F4E78;">Легенда та зведення за фільтром</h3>
    <div id="legend"></div>
    <table id="summaryTable">
      <thead><tr><th>Сектор</th><th>Кількість</th><th>Сер. інтенсивність</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = {json.dumps(events_json, ensure_ascii=False)};
const COLORS = {{
  "Артилерійський обстріл": "#A04030",
  "Активність БПЛА": "#1F4E78",
  "Штурмова дія": "#B87333",
  "Радіоелектронне подавлення": "#4B5320"
}};
const dates = [...new Set(DATA.map(d => d.date))].sort();
const types = [...new Set(DATA.map(d => d.event_type))];
const dateSel = document.getElementById('dateFilter');
dates.forEach(d => {{ const o = document.createElement('option'); o.value = d; o.textContent = d; dateSel.appendChild(o); }});
const typeSel = document.getElementById('typeFilter');
types.forEach(t => {{ const o = document.createElement('option'); o.value = t; o.textContent = t; typeSel.appendChild(o); }});
const leg = document.getElementById('legend');
Object.entries(COLORS).forEach(([k, c]) => {{
  leg.innerHTML += `<div style="margin-bottom:5px;font-size:12px;"><span class="badge" style="background:${{c}}"></span>${{k}}</div>`;
}});

const minLon = 31.55, maxLon = 32.95, minLat = 48.95, maxLat = 49.88;
function projX(lon) {{ return 55 + ((lon - minLon) / (maxLon - minLon)) * 660; }}
function projY(lat) {{ return 475 - ((lat - minLat) / (maxLat - minLat)) * 430; }}

function render() {{
  const dVal = dateSel.value;
  const tVal = typeSel.value;
  const iVal = parseFloat(document.getElementById('intFilter').value);
  document.getElementById('intVal').textContent = iVal.toFixed(1);
  const filtered = DATA.filter(r => (dVal === 'ALL' || r.date === dVal) && (tVal === 'ALL' || r.event_type === tVal) && r.intensity_index >= iVal);
  document.getElementById('counter').textContent = `Відображено подій: ${{filtered.length}} із ${{DATA.length}}`;
  const svg = document.getElementById('mapSvg');
  svg.innerHTML = '';
  for (let lon = 31.6; lon <= 32.9; lon += 0.2) {{
    const x = projX(lon);
    svg.innerHTML += `<line x1="${{x}}" y1="30" x2="${{x}}" y2="480" stroke="#CBD5E0" stroke-dasharray="3,3"/>`;
    svg.innerHTML += `<text x="${{x-14}}" y="500" font-size="11" fill="#4A5568">${{lon.toFixed(1)}}°E</text>`;
  }}
  for (let lat = 49.0; lat <= 49.8; lat += 0.2) {{
    const y = projY(lat);
    svg.innerHTML += `<line x1="50" y1="${{y}}" x2="725" y2="${{y}}" stroke="#CBD5E0" stroke-dasharray="3,3"/>`;
    svg.innerHTML += `<text x="8" y="${{y+4}}" font-size="11" fill="#4A5568">${{lat.toFixed(1)}}°N</text>`;
  }}
  const tip = document.getElementById('tooltip');
  filtered.forEach(r => {{
    const cx = projX(r.longitude), cy = projY(r.latitude), rad = Math.max(4, r.intensity_index * 1.45);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', rad);
    c.setAttribute('fill', COLORS[r.event_type] || '#1F4E78');
    c.setAttribute('fill-opacity', '0.78');
    c.setAttribute('stroke', '#ffffff'); c.setAttribute('stroke-width', '1');
    c.addEventListener('mousemove', (e) => {{
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY + 12) + 'px';
      tip.innerHTML = `<b>${{r.event_id}} — ${{r.event_type}}</b><br>Час: ${{r.timestamp}}<br>Сектор: ${{r.sector}}<br>Координати: ${{r.latitude}}°N, ${{r.longitude}}°E<br>Інтенсивність: ${{r.intensity_index}} балів (${{r.duration_min}} хв)`;
    }});
    c.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
    svg.appendChild(c);
  }});
  const secStats = {{}};
  filtered.forEach(r => {{
    if (!secStats[r.sector]) secStats[r.sector] = {{n: 0, s: 0}};
    secStats[r.sector].n += 1; secStats[r.sector].s += r.intensity_index;
  }});
  const tbody = document.querySelector('#summaryTable tbody');
  tbody.innerHTML = '';
  Object.entries(secStats).forEach(([sec, st]) => {{
    tbody.innerHTML += `<tr><td>${{sec}}</td><td>${{st.n}}</td><td>${{(st.s/st.n).toFixed(2)}}</td></tr>`;
  }});
}}
dateSel.addEventListener('change', render);
typeSel.addEventListener('change', render);
document.getElementById('intFilter').addEventListener('input', render);
render();
</script>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "n_raw": n_initial,
        "n_clean": len(df_clean),
        "n_dups": n_dups_removed,
        "n_swapped": n_swapped_fixed,
        "n_invalid": n_invalid_removed,
        "by_type": df_clean["event_type"].value_counts().to_dict(),
        "by_sector": df_clean.groupby("sector")["intensity_index"].agg(["count", "mean", "median"]).round(2).to_dict(orient="index"),
        "peak_date": df_clean.groupby("date").size().idxmax(),
        "peak_date_count": int(df_clean.groupby("date").size().max()),
        "mean_intensity": round(float(df_clean["intensity_index"].mean()), 2),
        "std_intensity": round(float(df_clean["intensity_index"].std()), 2),
    }


# ==============================================================================
# ВАРІАНТ 2: ДИНАМІКА АТАК (ЛІНІЙНИЙ ГРАФІК ТА ПОРІВНЯЛЬНА СТОВПЧИКОВА ДІАГРАМА)
# ==============================================================================
def run_variant_2(csv_dir, png_dir):
    rng = np.random.default_rng(SEED + 2)
    dates = pd.date_range("2026-10-01", "2026-10-30", freq="D")
    directions = ["Північний напрямок", "Східний напрямок", "Південний напрямок", "Західний напрямок"]
    base_rates = {
        "Північний напрямок": 14.0,
        "Східний напрямок": 24.5,
        "Південний напрямок": 18.0,
        "Західний напрямок": 9.5,
    }

    rows = []
    for i, d in enumerate(dates):
        wave = 7.5 * math.exp(-((i - 13) / 2.5) ** 2) + 5.0 * math.exp(-((i - 23) / 2.0) ** 2)
        for dir_name in directions:
            mult = 1.35 if dir_name == "Східний напрямок" else (1.1 if dir_name == "Південний напрямок" else 0.7)
            attacks = max(1, int(round(base_rates[dir_name] + mult * wave + rng.normal(0, 2.8))))
            repulsed = int(round(attacks * rng.uniform(0.76, 0.94)))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "week_num": f"Тиждень {(i // 7) + 1}",
                "direction": dir_name,
                "attacks_per_day": attacks,
                "repulsed_attacks": repulsed,
                "artillery_salvos": int(attacks * rng.uniform(4.2, 6.8)),
                "is_synthetic_training_data": True,
            })

    df_raw = pd.DataFrame(rows)
    df_raw = pd.concat([df_raw, df_raw.iloc[[4, 29]]], ignore_index=True)
    df_raw.loc[11, "attacks_per_day"] = -5
    df_raw.loc[47, "attacks_per_day"] = np.nan
    df_raw.to_csv(os.path.join(csv_dir, "v2_attacks_dynamics_raw.csv"), index=False, encoding="utf-8")

    df_clean = df_raw.drop_duplicates(subset=["date", "direction"]).copy()
    bad_mask = df_clean["attacks_per_day"].isna() | (df_clean["attacks_per_day"] < 0)
    n_bad_v2 = int(bad_mask.sum())
    df_clean = df_clean[~bad_mask].reset_index(drop=True)
    df_clean.to_csv(os.path.join(csv_dir, "v2_attacks_dynamics_clean.csv"), index=False, encoding="utf-8")

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(14.2, 5.9), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )
    dir_colors = {
        "Східний напрямок": PALETTE["earth_terracotta"],
        "Південний напрямок": PALETTE["nato_blue"],
        "Північний напрямок": PALETTE["olive_dark"],
        "Західний напрямок": PALETTE["earth_ochre"],
    }
    dir_markers = {
        "Східний напрямок": "o",
        "Південний напрямок": "s",
        "Північний напрямок": "^",
        "Західний напрямок": "d",
    }

    df_clean["date_dt"] = pd.to_datetime(df_clean["date"])
    for d_name in directions:
        sub = df_clean[df_clean["direction"] == d_name].sort_values("date_dt")
        ax_line.plot(
            sub["date_dt"],
            sub["attacks_per_day"],
            label=d_name,
            color=dir_colors[d_name],
            marker=dir_markers[d_name],
            markersize=4,
            linewidth=1.8,
            alpha=0.9,
        )

    ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_line.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax_line.set_title("Динаміка добової кількості умовних атак за напрямками", fontsize=11, fontweight="bold")
    ax_line.set_xlabel("Дата спостереження (жовтень 2026 р.)")
    ax_line.set_ylabel("Інтенсивність дій, кількість атак / добу")
    ax_line.legend(loc="upper left", fontsize=8.5)

    agg_df = (
        df_clean.groupby("direction")["attacks_per_day"]
        .agg(["count", "sum", "mean", "median", "std", "max"])
        .loc[directions]
    )

    x_pos = np.arange(len(directions))
    width = 0.36
    bars_mean = ax_bar.bar(
        x_pos - width / 2,
        agg_df["mean"],
        width,
        yerr=agg_df["std"],
        capsize=4,
        color=PALETTE["nato_blue"],
        label="Середнє арифметичне (Mean ± SD), атак/добу",
        alpha=0.9,
    )
    bars_med = ax_bar.bar(
        x_pos + width / 2,
        agg_df["median"],
        width,
        color=PALETTE["olive_mid"],
        label="Медіанне значення (Median), атак/добу",
        alpha=0.9,
    )

    for idx_b, b in enumerate(bars_mean):
        sd_val = agg_df["std"].iloc[idx_b]
        ax_bar.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + sd_val + 0.6,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    for b in bars_med:
        ax_bar.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([d.replace(" напрямок", "\nнапрямок") for d in directions], fontsize=8.5)
    ax_bar.set_title(
        "Порівняння напрямків за 30 діб\n(Агрегування: середнє за добу ± СКВ та медіана)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_bar.set_ylabel("Агрегована кількість атак, од./добу")
    ax_bar.set_ylim(0, agg_df["mean"].max() + agg_df["std"].max() + 6.5)
    ax_bar.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Варіант 2. Добова динаміка та порівняльна агрегація атак за напрямками (Навчальні дані, seed=42)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v2_attacks_dynamics.png"))
    plt.close(fig)

    return {
        "n_clean": len(df_clean),
        "n_bad_removed": n_bad_v2,
        "agg_stats": agg_df.round(2).to_dict(orient="index"),
        "total_attacks": int(df_clean["attacks_per_day"].sum()),
        "peak_row": df_clean.loc[df_clean["attacks_per_day"].idxmax()][["date", "direction", "attacks_per_day"]].to_dict(),
    }


# ==============================================================================
# ВАРІАНТ 3: СТРУКТУРА СИЛ (TREEMAP ТА КРУГОВА ДІАГРАМА ЗІСТАВНИХ ПОКАЗНИКІВ)
# ==============================================================================
def run_variant_3(csv_dir, png_dir):
    rows = [
        {"record_id": "FS-01", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Механізовані підрозділи", "subcategory": "Піхотні роти", "quantity": 1420},
        {"record_id": "FS-02", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Механізовані підрозділи", "subcategory": "Танкові екіпажі", "quantity": 310},
        {"record_id": "FS-03", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Артилерійські підрозділи", "subcategory": "Ствольна артилерія", "quantity": 540},
        {"record_id": "FS-04", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Артилерійські підрозділи", "subcategory": "Реактивна артилерія", "quantity": 210},
        {"record_id": "FS-05", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Підрозділи БПЛА та розвідки", "subcategory": "Екіпажі розвідувальних БПЛА", "quantity": 280},
        {"record_id": "FS-06", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Підрозділи БПЛА та розвідки", "subcategory": "Екіпажі ударних БПЛА", "quantity": 340},
        {"record_id": "FS-07", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Підрозділи забезпечення", "subcategory": "Інженерно-саперні групи", "quantity": 390},
        {"record_id": "FS-08", "accounting_domain": "Особовий склад", "unit_of_measure": "осіб", "branch": "Підрозділи забезпечення", "subcategory": "Логістика, зв'язок та РЕБ", "quantity": 510},
        {"record_id": "FS-09", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Бойові броньовані машини", "subcategory": "БМП та БТР", "quantity": 138},
        {"record_id": "FS-10", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Бойові броньовані машини", "subcategory": "Бронеавтомобілі", "quantity": 94},
        {"record_id": "FS-11", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Танки", "subcategory": "Основні бойові танки", "quantity": 62},
        {"record_id": "FS-12", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Артилерійські системи", "subcategory": "Самохідні та причіпні гаубиці", "quantity": 84},
        {"record_id": "FS-13", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Артилерійські системи", "subcategory": "РСЗВ та міномети", "quantity": 52},
        {"record_id": "FS-14", "accounting_domain": "Озброєння та техніка", "unit_of_measure": "од. техніки", "branch": "Спеціальна та інженерна техніка", "subcategory": "Засоби РЕБ, зв'язку та тягачі", "quantity": 70},
        {"record_id": "FS-ERR", "accounting_domain": "Змішана категорія (помилка)", "unit_of_measure": "невизначено", "branch": "Зведена група", "subcategory": "Люди + техніка разом", "quantity": -15},
    ]
    df_raw = pd.DataFrame(rows)
    df_raw["is_synthetic_training_data"] = True
    df_raw.to_csv(os.path.join(csv_dir, "v3_force_structure_raw.csv"), index=False, encoding="utf-8")

    df_clean = df_raw[df_raw["quantity"] > 0].copy()
    df_clean["domain_total"] = df_clean.groupby("accounting_domain")["quantity"].transform("sum")
    df_clean["share_pct"] = (df_clean["quantity"] / df_clean["domain_total"] * 100.0).round(2)
    df_clean.to_csv(os.path.join(csv_dir, "v3_force_structure_clean.csv"), index=False, encoding="utf-8")

    df_pers = df_clean[df_clean["accounting_domain"] == "Особовий склад"].sort_values("quantity", ascending=False)
    df_eq = df_clean[df_clean["accounting_domain"] == "Озброєння та техніка"].sort_values("quantity", ascending=False)

    fig, (ax_tree, ax_pie) = plt.subplots(1, 2, figsize=(14.4, 6.3))

    rects = squarify_rects(df_pers["quantity"].values, 0, 0, 100, 100)
    branch_colors = {
        "Механізовані підрозділи": PALETTE["olive_dark"],
        "Артилерійські підрозділи": PALETTE["nato_blue"],
        "Підрозділи БПЛА та розвідки": PALETTE["earth_ochre"],
        "Підрозділи забезпечення": PALETTE["slate_grey"],
    }
    short_sub = {
        "Екіпажі розвідувальних БПЛА": "Екіпажі розвід.\nБПЛА",
        "Екіпажі ударних БПЛА": "Екіпажі ударних\nБПЛА",
        "Логістика, зв'язок та РЕБ": "Логістика,\nзв'язок та РЕБ",
        "Інженерно-саперні групи": "Інженерно-\nсаперні групи",
        "Реактивна артилерія": "Реактивна\nартилерія",
    }
    for (_, row), (rx, ry, rw, rh) in zip(df_pers.iterrows(), rects):
        col = branch_colors.get(row["branch"], PALETTE["olive_mid"])
        rect_patch = patches.Rectangle(
            (rx, ry), rw, rh, facecolor=col, edgecolor="white", linewidth=2.2, alpha=0.88
        )
        ax_tree.add_patch(rect_patch)
        sname = short_sub.get(row["subcategory"], row["subcategory"])
        label_txt = f"{sname}\n{row['quantity']} осіб ({row['share_pct']:.1f}%)"
        ax_tree.text(
            rx + rw / 2,
            ry + rh / 2,
            label_txt,
            ha="center",
            va="center",
            color="white",
            fontsize=8.2,
            fontweight="bold",
        )
    ax_tree.set_xlim(0, 100)
    ax_tree.set_ylim(0, 100)
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    ax_tree.set_title(
        f"Treemap структури особового складу\n(Одиниця обліку: осіб, усього N = {int(df_pers['quantity'].sum())} осіб)",
        fontsize=11,
        fontweight="bold",
    )

    leg_patches = [patches.Patch(color=c, label=b) for b, c in branch_colors.items()]
    ax_tree.legend(handles=leg_patches, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5)

    eq_colors = [
        PALETTE["nato_blue"],
        "#3B6E9C",
        PALETTE["olive_dark"],
        PALETTE["earth_ochre"],
        PALETTE["earth_terracotta"],
        PALETTE["slate_grey"],
    ]
    wedges, texts, autotexts = ax_pie.pie(
        df_eq["quantity"],
        labels=[f"{r['subcategory']}\n({r['quantity']} од.)" for _, r in df_eq.iterrows()],
        autopct="%1.1f%%",
        startangle=135,
        colors=eq_colors,
        pctdistance=0.76,
        wedgeprops=dict(width=0.46, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=8.5),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8.5)
    ax_pie.set_title(
        f"Структура парку озброєння та військової техніки\n(Одиниця обліку: од. техніки, усього M = {int(df_eq['quantity'].sum())} од.)",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "Варіант 3. Структура умовного угруповання за роздільними одиницями обліку (Навчальні дані)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v3_force_structure.png"))
    plt.close(fig)

    return {
        "personnel_total": int(df_pers["quantity"].sum()),
        "equipment_total": int(df_eq["quantity"].sum()),
        "personnel_breakdown": df_pers[["branch", "subcategory", "quantity", "share_pct"]].to_dict(orient="records"),
        "equipment_breakdown": df_eq[["branch", "subcategory", "quantity", "share_pct"]].to_dict(orient="records"),
    }


# ==============================================================================
# ВАРІАНТ 4: ПОРІВНЯННЯ БОЄЗДАТНОСТІ (РАДАРНА ДІАГРАМА ЗА 5 ПАРАМЕТРАМИ)
# ==============================================================================
def run_variant_4(csv_dir, png_dir):
    raw_units = [
        {
            "unit_code": "ОМБр-А («Північ»)",
            "staffing_pct_raw": 92.0,
            "equip_serviceable_pct_raw": 86.0,
            "ammo_days_raw": 5.4,
            "training_hours_raw": 148.0,
            "c2_readiness_index_raw": 88.0,
        },
        {
            "unit_code": "ОМБр-Б («Схід»)",
            "staffing_pct_raw": 84.0,
            "equip_serviceable_pct_raw": 79.0,
            "ammo_days_raw": 5.8,
            "training_hours_raw": 152.0,
            "c2_readiness_index_raw": 91.0,
        },
        {
            "unit_code": "ОШБ-В («Південь»)",
            "staffing_pct_raw": 95.0,
            "equip_serviceable_pct_raw": 91.0,
            "ammo_days_raw": 4.2,
            "training_hours_raw": 156.0,
            "c2_readiness_index_raw": 84.0,
        },
        {
            "unit_code": "БТрО-Г («Резерв»)",
            "staffing_pct_raw": 89.0,
            "equip_serviceable_pct_raw": 72.0,
            "ammo_days_raw": 4.8,
            "training_hours_raw": 118.0,
            "c2_readiness_index_raw": 76.0,
        },
    ]
    df_raw = pd.DataFrame(raw_units)
    df_raw["is_synthetic_training_data"] = True
    df_raw.to_csv(os.path.join(csv_dir, "v4_combat_capability_raw.csv"), index=False, encoding="utf-8")

    df_clean = df_raw.copy()
    df_clean["p1_staffing_score"] = (df_clean["staffing_pct_raw"] / 100.0 * 100.0).round(1)
    df_clean["p2_equipment_score"] = (df_clean["equip_serviceable_pct_raw"] / 100.0 * 100.0).round(1)
    df_clean["p3_ammo_score"] = (df_clean["ammo_days_raw"] / 6.0 * 100.0).clip(0, 100).round(1)
    df_clean["p4_training_score"] = (df_clean["training_hours_raw"] / 160.0 * 100.0).clip(0, 100).round(1)
    df_clean["p5_c2_score"] = (df_clean["c2_readiness_index_raw"] / 100.0 * 100.0).round(1)

    score_cols = ["p1_staffing_score", "p2_equipment_score", "p3_ammo_score", "p4_training_score", "p5_c2_score"]
    df_clean["integral_mean_score"] = df_clean[score_cols].mean(axis=1).round(2)
    df_clean.to_csv(os.path.join(csv_dir, "v4_combat_capability_clean.csv"), index=False, encoding="utf-8")

    categories = [
        "Укомплектованість\nо/с (P1)",
        "Справність\nОВТ (P2)",
        "Забезпеченість\nбоєприпасами (P3)",
        "Рівень бойової\nпідготовки (P4)",
        "Стійкість управління\nта зв'язку (P5)",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(14.2, 6.2))
    ax_rad = fig.add_subplot(121, polar=True)
    ax_tbl = fig.add_subplot(122)

    unit_colors = [PALETTE["nato_blue"], PALETTE["earth_terracotta"], PALETTE["olive_dark"], PALETTE["earth_ochre"]]
    unit_styles = ["-", "--", "-.", ":"]

    ax_rad.set_theta_offset(np.pi / 2)
    ax_rad.set_theta_direction(-1)
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax_rad.set_rlabel_position(18)
    ax_rad.set_yticks([20, 40, 60, 80, 100])
    ax_rad.set_yticklabels(["20", "40", "60", "80", "100 балів"], fontsize=8, color=PALETTE["slate_grey"])
    ax_rad.set_ylim(0, 105)

    for idx, row in df_clean.iterrows():
        vals = [row[c] for c in score_cols]
        vals += vals[:1]
        ax_rad.plot(
            angles,
            vals,
            color=unit_colors[idx],
            linestyle=unit_styles[idx],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=f"{row['unit_code']} (Інт.: {row['integral_mean_score']:.1f})",
        )
        ax_rad.fill(angles, vals, color=unit_colors[idx], alpha=0.10)

    ax_rad.set_title("Радарна діаграма нормалізованих параметрів (0–100 балів)", fontsize=11, fontweight="bold", pad=22)
    ax_rad.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.5)

    ax_tbl.axis("off")
    table_data = []
    for _, row in df_clean.iterrows():
        table_data.append([
            row["unit_code"],
            f"{row['p1_staffing_score']:.1f}",
            f"{row['p2_equipment_score']:.1f}",
            f"{row['p3_ammo_score']:.1f}\n({row['ammo_days_raw']} БК)",
            f"{row['p4_training_score']:.1f}\n({int(row['training_hours_raw'])} год)",
            f"{row['p5_c2_score']:.1f}",
            f"{row['integral_mean_score']:.2f}",
        ])
    col_labels = ["Підрозділ", "P1:\nО/С", "P2:\nОВТ", "P3: БК\n(норм.)", "P4: Підг.\n(норм.)", "P5:\nУпр.", "Середній\nбал"]
    col_widths = [0.25, 0.10, 0.10, 0.14, 0.15, 0.10, 0.14]
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.3)
    tbl.scale(1.02, 2.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(PALETTE["nato_blue"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 6:
            cell.set_facecolor("#EBF1F5")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if r % 2 == 1 else "#F7F6F2")

    ax_tbl.set_title(
        "Таблиця нормалізованих значень за параметрами P1–P5\n(Нормалізація: P3 відносно 6.0 БК = 100 б., P4 відносно 160 год = 100 б.)",
        fontsize=10.5,
        fontweight="bold",
    )

    fig.suptitle(
        "Варіант 4. Багатовимірне порівняння умовної боєздатності підрозділів (Навчальні дані)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v4_combat_capability_radar.png"))
    plt.close(fig)

    return {
        "units": df_clean.to_dict(orient="records"),
        "leader": df_clean.loc[df_clean["integral_mean_score"].idxmax()]["unit_code"],
        "leader_score": float(df_clean["integral_mean_score"].max()),
    }


# ==============================================================================
# ВАРІАНТ 5: МЕРЕЖЕВА ВЗАЄМОДІЯ (ОРІЄНТОВАНИЙ ГРАФ ТА ЦЕНТРАЛЬНОСТІ)
# ==============================================================================
def run_variant_5(csv_dir, png_dir):
    nodes = [
        "КП ОТУ",
        "Ретранслятор-1",
        "Вузол розвідки",
        "ЦУ БПЛА",
        "КП Бр-1",
        "КП Бр-2",
        "КП Артгрупи",
        "Підрозділ РЕБ",
        "Логістичний центр",
        "Передовий ПУ-1",
        "Передовий ПУ-2",
    ]
    edges = [
        ("Вузол розвідки", "Ретранслятор-1", "Розвіддані", 45, 120),
        ("Вузол розвідки", "ЦУ БПЛА", "Розвіддані", 50, 95),
        ("ЦУ БПЛА", "Ретранслятор-1", "Розвіддані", 60, 140),
        ("ЦУ БПЛА", "КП Артгрупи", "Цілевказання", 40, 110),
        ("Ретранслятор-1", "КП ОТУ", "Розвіддані", 100, 260),
        ("КП ОТУ", "Ретранслятор-1", "Бойове розпорядження", 80, 180),
        ("Ретранслятор-1", "КП Бр-1", "Бойове розпорядження", 50, 90),
        ("Ретранслятор-1", "КП Бр-2", "Бойове розпорядження", 50, 85),
        ("КП ОТУ", "КП Артгрупи", "Бойове розпорядження", 60, 75),
        ("КП ОТУ", "Підрозділ РЕБ", "Бойове розпорядження", 35, 45),
        ("КП ОТУ", "Логістичний центр", "Запит логістики", 40, 50),
        ("КП Бр-1", "Передовий ПУ-1", "Бойове розпорядження", 25, 65),
        ("КП Бр-2", "Передовий ПУ-2", "Бойове розпорядження", 25, 60),
        ("Передовий ПУ-1", "КП Бр-1", "Розвіддані", 20, 70),
        ("Передовий ПУ-2", "КП Бр-2", "Розвіддані", 20, 68),
        ("КП Бр-1", "КП Артгрупи", "Цілевказання", 30, 55),
        ("КП Бр-2", "КП Артгрупи", "Цілевказання", 30, 52),
        ("КП Бр-1", "Логістичний центр", "Запит логістики", 25, 40),
        ("КП Бр-2", "Логістичний центр", "Запит логістики", 25, 38),
        ("Логістичний центр", "КП Бр-1", "Підтвердження постачання", 25, 35),
        ("Логістичний центр", "КП Бр-2", "Підтвердження постачання", 25, 34),
        ("Підрозділ РЕБ", "Вузол розвідки", "Розвіддані", 30, 60),
        ("КП ОТУ", "КП ОТУ", "Помилкова петля", 0, 0),
        ("Вузол розвідки", "Ретранслятор-1", "Розвіддані", 45, 120),
    ]

    df_edges_raw = pd.DataFrame(
        edges, columns=["source", "target", "link_type", "bandwidth_mbps", "msg_per_hour"]
    )
    df_edges_raw["is_synthetic_training_data"] = True
    df_edges_raw.to_csv(os.path.join(csv_dir, "v5_network_edges_raw.csv"), index=False, encoding="utf-8")

    df_edges_clean = df_edges_raw.drop_duplicates(subset=["source", "target", "link_type"]).copy()
    df_edges_clean = df_edges_clean[df_edges_clean["source"] != df_edges_clean["target"]].reset_index(drop=True)
    df_edges_clean.to_csv(os.path.join(csv_dir, "v5_network_edges_clean.csv"), index=False, encoding="utf-8")

    clean_tuples = list(df_edges_clean[["source", "target", "link_type", "bandwidth_mbps", "msg_per_hour"]].itertuples(index=False, name=None))
    df_cent = compute_directed_centralities(nodes, clean_tuples)
    df_cent = df_cent.sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)
    df_cent.to_csv(os.path.join(csv_dir, "v5_network_centralities.csv"), index=False, encoding="utf-8")

    pos = {
        "КП ОТУ": (0.50, 0.90),
        "Ретранслятор-1": (0.50, 0.58),
        "Вузол розвідки": (0.16, 0.80),
        "Підрозділ РЕБ": (0.14, 0.48),
        "ЦУ БПЛА": (0.32, 0.70),
        "КП Артгрупи": (0.68, 0.70),
        "Логістичний центр": (0.86, 0.52),
        "КП Бр-1": (0.34, 0.34),
        "КП Бр-2": (0.66, 0.34),
        "Передовий ПУ-1": (0.30, 0.12),
        "Передовий ПУ-2": (0.74, 0.12),
    }

    link_colors = {
        "Бойове розпорядження": PALETTE["nato_blue"],
        "Розвіддані": PALETTE["olive_dark"],
        "Цілевказання": PALETTE["earth_terracotta"],
        "Запит логістики": PALETTE["earth_ochre"],
        "Підтвердження постачання": PALETTE["slate_grey"],
    }

    fig, (ax_net, ax_bar) = plt.subplots(
        1, 2, figsize=(14.6, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    for u, v, ltype, bw, mph in clean_tuples:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        rad = 0.14 if any((u2 == v and v2 == u) for u2, v2, *_ in clean_tuples) else 0.04
        arrow = patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=7,head_width=4.5",
            color=link_colors.get(ltype, PALETTE["slate_grey"]),
            linewidth=1.6,
            alpha=0.82,
            shrinkA=18,
            shrinkB=18,
            zorder=2,
        )
        ax_net.add_patch(arrow)

    bet_map = dict(zip(df_cent["node_id"], df_cent["betweenness_centrality"]))
    for node, (nx_coord, ny_coord) in pos.items():
        b_val = bet_map[node]
        size = 420 + b_val * 2200
        ax_net.scatter(
            nx_coord,
            ny_coord,
            s=size,
            c=PALETTE["nato_blue"] if b_val < 0.25 else PALETTE["earth_terracotta"],
            edgecolors="white",
            linewidth=2.0,
            zorder=4,
        )
        ax_net.text(
            nx_coord,
            ny_coord - 0.06,
            f"{node}\n(C_B={b_val:.2f})",
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D8D6CE", alpha=0.92),
            zorder=5,
        )

    ax_net.set_xlim(0.02, 0.98)
    ax_net.set_ylim(-0.02, 1.02)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_title(
        "Орієнтований граф інформаційного обміну\n(Розмір вершини ~ посередництво C_B; колір стрілки = тип зв'язку)",
        fontsize=10.5,
        fontweight="bold",
    )
    leg_lines = [Line2D([0], [0], color=c, lw=2.2, label=k) for k, c in link_colors.items()]
    ax_net.legend(
        handles=leg_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=8,
        title="Тип інформаційного потоку",
    )

    y_pos = np.arange(len(df_cent))
    h = 0.27
    ax_bar.barh(
        y_pos - h,
        df_cent["betweenness_centrality"],
        height=h,
        color=PALETTE["earth_terracotta"],
        label="Betweenness (Посередництво C_B)",
    )
    ax_bar.barh(
        y_pos,
        df_cent["in_degree_centrality"],
        height=h,
        color=PALETTE["nato_blue"],
        label="In-Degree (Вхідна центральність C_in)",
    )
    ax_bar.barh(
        y_pos + h,
        df_cent["out_degree_centrality"],
        height=h,
        color=PALETTE["olive_dark"],
        label="Out-Degree (Вихідна центральність C_out)",
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(df_cent["node_id"], fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Нормалізоване значення показника центральності [0..1]")
    ax_bar.set_title("Показники центральності вузлів мережі управління", fontsize=10.5, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Варіант 5. Аналіз мережевої взаємодії та топологічної центральності вузлів (Навчальні дані)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v5_network_interaction.png"))
    plt.close(fig)

    return {
        "n_nodes": len(nodes),
        "n_edges": len(df_edges_clean),
        "centralities": df_cent.to_dict(orient="records"),
    }


# ==============================================================================
# ВАРІАНТ 6: КОРЕЛЯЦІЙНИЙ АНАЛІЗ (ТЕПЛОВА КАРТА, ЦИКЛІЧНИЙ ЧАС ТА ЗМІННА ВТРАТ)
# ==============================================================================
def run_variant_6(csv_dir, png_dir):
    rng = np.random.default_rng(SEED + 6)
    n_obs = 120
    hour_of_day = rng.uniform(0, 24, size=n_obs).round(2)
    time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

    terrain_nominal = rng.choice(["Лісостеп", "Степ", "Урбанізована зона"], size=n_obs, p=[0.4, 0.4, 0.2])

    recon_uav_flights = np.clip(np.round(12 + 6 * (-time_cos) + rng.normal(0, 3.2, n_obs)), 1, 30)
    artillery_intensity = np.clip(np.round(25 + 1.45 * recon_uav_flights + 4.5 * time_sin + rng.normal(0, 5.5, n_obs)), 2, 85)
    ew_suppression_db = np.clip(np.round(40 + 0.35 * recon_uav_flights + rng.normal(0, 6.5, n_obs), 1), 15, 75)
    response_delay_min = np.clip(np.round(28 - 0.32 * ew_suppression_db + rng.normal(0, 3.8, n_obs), 1), 4, 45)
    fortification_level = np.clip(np.round(rng.uniform(2.0, 9.5, n_obs), 1), 1.0, 10.0)

    conditional_losses_units = np.clip(
        np.round(
            2.2
            + 0.085 * artillery_intensity
            + 0.14 * response_delay_min
            - 0.38 * fortification_level
            + rng.normal(0, 0.9, n_obs),
            2,
        ),
        0.0,
        15.0,
    )

    df_raw = pd.DataFrame({
        "episode_id": [f"EP-{i:03d}" for i in range(1, n_obs + 1)],
        "terrain_nominal": terrain_nominal,
        "hour_of_day": hour_of_day,
        "time_sin": np.round(time_sin, 4),
        "time_cos": np.round(time_cos, 4),
        "recon_uav_flights": recon_uav_flights,
        "artillery_intensity": artillery_intensity,
        "ew_suppression_db": ew_suppression_db,
        "response_delay_min": response_delay_min,
        "fortification_level": fortification_level,
        "conditional_losses_units": conditional_losses_units,
        "is_synthetic_training_data": True,
    })
    df_raw.loc[[7, 53], "artillery_intensity"] = np.nan
    df_raw.to_csv(os.path.join(csv_dir, "v6_correlation_raw.csv"), index=False, encoding="utf-8")

    df_clean = df_raw.dropna().reset_index(drop=True)
    df_clean.to_csv(os.path.join(csv_dir, "v6_correlation_clean.csv"), index=False, encoding="utf-8")

    num_cols = [
        "time_sin",
        "time_cos",
        "recon_uav_flights",
        "artillery_intensity",
        "ew_suppression_db",
        "response_delay_min",
        "fortification_level",
        "conditional_losses_units",
    ]
    labels_uk = [
        "Час доби: sin(2πt/24)",
        "Час доби: cos(2πt/24)",
        "Вильоти розвід. БПЛА (од.)",
        "Інтенсивність обстрілів (залпів)",
        "Рівень РЕБ-подавлення (дБ)",
        "Затримка реакції (хв)",
        "Інженерне обладнання (бал)",
        "Умовні втрати (од.)",
    ]

    corr_pearson = df_clean[num_cols].corr(method="pearson")
    corr_spearman = df_clean[num_cols].corr(method="spearman")

    corr_pearson.index = labels_uk
    corr_pearson.columns = labels_uk
    corr_spearman.index = labels_uk
    corr_spearman.columns = labels_uk

    corr_pearson.to_csv(os.path.join(csv_dir, "v6_matrix_pearson.csv"), encoding="utf-8")
    corr_spearman.to_csv(os.path.join(csv_dir, "v6_matrix_spearman.csv"), encoding="utf-8")

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(15.0, 6.4))
    cmap_div = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_pearson,
        ax=ax_p,
        cmap=cmap_div,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        cbar_kws={"shrink": 0.82, "label": "Коефіцієнт r Пірсона"},
        annot_kws={"size": 8.5, "weight": "bold"},
    )
    ax_p.set_title("Матриця лінійної кореляції Пірсона (r)\n(Без номінальних категорій; час доби через sin/cos)", fontsize=10.5, fontweight="bold")
    ax_p.tick_params(axis="x", rotation=32)

    sns.heatmap(
        corr_spearman,
        ax=ax_s,
        cmap=cmap_div,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        cbar_kws={"shrink": 0.82, "label": "Коефіцієнт ρ Спірмена"},
        annot_kws={"size": 8.5, "weight": "bold"},
    )
    ax_s.set_title("Матриця рангової кореляції Спірмена (ρ)\n(Стійка до викидів та нелінійних монотонних залежностей)", fontsize=10.5, fontweight="bold")
    ax_s.tick_params(axis="x", rotation=32)

    fig.suptitle(
        "Варіант 6. Кореляційний аналіз факторів обстановки та змінної умовних втрат (n = 118 епізодів, seed=42)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v6_correlation_heatmap.png"))
    plt.close(fig)

    loss_corrs = {}
    for col, lbl in zip(num_cols[:-1], labels_uk[:-1]):
        r_val, p_val = stats.pearsonr(df_clean[col], df_clean["conditional_losses_units"])
        rho_val, sp_pval = stats.spearmanr(df_clean[col], df_clean["conditional_losses_units"])
        loss_corrs[lbl] = {
            "pearson_r": round(float(r_val), 3),
            "pearson_p": float(f"{p_val:.4e}"),
            "spearman_rho": round(float(rho_val), 3),
            "spearman_p": float(f"{sp_pval:.4e}"),
        }

    return {
        "n_clean": len(df_clean),
        "loss_correlations": loss_corrs,
    }


# ==============================================================================
# ВАРІАНТ 7: ІНТЕРАКТИВНИЙ ДАШБОРД (4 УЗГОДЖЕНІ ВІЗУАЛІЗАЦІЇ + ФІЛЬТРИ + ТАБЛИЦЯ)
# ==============================================================================
def run_variant_7(csv_dir, png_dir, html_dir):
    df_ev = pd.read_csv(os.path.join(csv_dir, "v1_events_clean.csv"))

    dash_df = (
        df_ev.groupby(["date", "sector", "event_type"])
        .agg(
            events_count=("event_id", "count"),
            mean_intensity=("intensity_index", "mean"),
            total_duration_min=("duration_min", "sum"),
        )
        .reset_index()
    )
    dash_df["mean_intensity"] = dash_df["mean_intensity"].round(2)
    dash_df["is_synthetic_training_data"] = True
    dash_df.to_csv(os.path.join(csv_dir, "v7_dashboard_dataset.csv"), index=False, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    ax1, ax2, ax3, ax4 = axes.flatten()

    d_sum = dash_df.groupby("date")["events_count"].sum()
    ax1.plot(range(len(d_sum)), d_sum.values, color=PALETTE["nato_blue"], marker="o", linewidth=2.2)
    ax1.fill_between(range(len(d_sum)), d_sum.values, color=PALETTE["nato_blue"], alpha=0.15)
    ax1.set_xticks(range(0, len(d_sum), 2))
    ax1.set_xticklabels([d[5:] for d in d_sum.index[::2]])
    ax1.set_title("Панель А: Добова динаміка кількості подій (од./добу)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Кількість подій, од.")

    sec_piv = dash_df.pivot_table(index="sector", columns="event_type", values="events_count", aggfunc="sum", fill_value=0)
    sec_piv.plot(
        kind="barh",
        stacked=True,
        ax=ax2,
        color=[PALETTE["earth_terracotta"], PALETTE["nato_blue"], PALETTE["olive_dark"], PALETTE["earth_ochre"]],
    )
    ax2.set_title("Панель Б: Розподіл подій за секторами та типами (од.)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Сумарна кількість подій, од.")
    ax2.set_ylabel("Сектор відповідальності")
    ax2.legend(fontsize=7.5, loc="lower right")

    sec_int = dash_df.groupby("sector").agg(
        mean_int=("mean_intensity", "mean"), total_dur=("total_duration_min", "sum")
    )
    bars = ax3.bar(sec_int.index, sec_int["mean_int"], color=PALETTE["olive_dark"], width=0.52)
    for b in bars:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0, 8.5)
    ax3.set_title("Панель В: Середній індекс інтенсивності за секторами (бали)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Середня інтенсивність, бали [1..10]")
    ax3.tick_params(axis="x", rotation=15)

    ax4.axis("off")
    tbl_rows = []
    for sec, row in sec_int.iterrows():
        cnt = int(dash_df[dash_df["sector"] == sec]["events_count"].sum())
        tbl_rows.append([sec, f"{cnt}", f"{row['mean_int']:.2f} б.", f"{int(row['total_dur'])} хв"])
    tbl = ax4.table(
        cellText=tbl_rows,
        colLabels=["Сектор", "Кількість подій", "Сер. інтенсивність", "Сумарна тривалість"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(PALETTE["nato_blue"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    ax4.set_title("Панель Г: Інтерактивна зведена таблиця (синхронізована з фільтрами)", fontsize=10.5, fontweight="bold")

    fig.suptitle(
        "Варіант 7. Компонування аналітичного дашборду ІАЗ (4 узгоджені панелі, навчальні дані)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v7_dashboard_overview.png"))
    plt.close(fig)

    html_dash_path = os.path.join(html_dir, "v7_interactive_dashboard.html")
    records_json = dash_df.to_dict(orient="records")
    html_dash = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Варіант 7 — Інтерактивний аналітичний дашборд ІАЗ</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 18px 24px; background: #F7F6F2; color: #1C2321; }}
  header {{ background: #1F4E78; color: #fff; padding: 14px 20px; margin-bottom: 16px; }}
  header h1 {{ margin: 0; font-size: 18px; }}
  header p {{ margin: 4px 0 0 0; font-size: 12px; opacity: 0.9; }}
  .filters {{ display: flex; gap: 16px; background: #fff; padding: 12px 16px; border: 1px solid #D8D6CE; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
  .kpi {{ background: #fff; border-top: 4px solid #4B5320; padding: 10px 14px; border-left: 1px solid #D8D6CE; border-right: 1px solid #D8D6CE; border-bottom: 1px solid #D8D6CE; }}
  .kpi .val {{ font-size: 22px; font-weight: bold; color: #1F4E78; }}
  .kpi .lbl {{ font-size: 12px; color: #4A5568; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .panel {{ background: #fff; border: 1px solid #D8D6CE; padding: 14px; }}
  .panel h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #1F4E78; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #E2E8F0; text-align: left; }}
  th {{ color: #1F4E78; border-bottom: 2px solid #1F4E78; }}
</style>
</head>
<body>
<header>
  <h1>Варіант 7. Інтерактивний дашборд моніторингу умовних подій (Синтетичні навчальні дані, seed=42)</h1>
  <p>Усі 4 візуалізації та таблиця синхронно оновлюються при зміні фільтрів сектора, типу події та діапазону дат.</p>
</header>
<div class="filters">
  <label><b>Сектор:</b> <select id="fSector"><option value="ALL">Усі сектори</option></select></label>
  <label><b>Тип події:</b> <select id="fType"><option value="ALL">Усі типи</option></select></label>
  <label><b>Дата від:</b> <select id="fDateStart"></select></label>
  <label><b>Дата до:</b> <select id="fDateEnd"></select></label>
</div>
<div class="kpi-row">
  <div class="kpi"><div class="lbl">Сумарна кількість подій</div><div class="val" id="kpiCount">0</div></div>
  <div class="kpi"><div class="lbl">Середня інтенсивність (бали)</div><div class="val" id="kpiInt">0.0</div></div>
  <div class="kpi"><div class="lbl">Сумарна тривалість (хв)</div><div class="val" id="kpiDur">0</div></div>
  <div class="kpi"><div class="lbl">Активних секторів у вибірці</div><div class="val" id="kpiSec">0</div></div>
</div>
<div class="grid">
  <div class="panel"><h3>1. Добова динаміка кількості подій (од./добу)</h3><svg id="svgLine" viewBox="0 0 520 220" width="100%"></svg></div>
  <div class="panel"><h3>2. Розподіл кількості подій за секторами (од.)</h3><svg id="svgSec" viewBox="0 0 520 220" width="100%"></svg></div>
  <div class="panel"><h3>3. Структура за типами подій (%)</h3><svg id="svgType" viewBox="0 0 520 220" width="100%"></svg></div>
  <div class="panel"><h3>4. Деталізована таблиця спостережень за фільтром</h3>
    <div style="max-height:210px;overflow-y:auto;">
      <table id="dataTable"><thead><tr><th>Дата</th><th>Сектор</th><th>Тип події</th><th>К-сть</th><th>Сер. інт.</th><th>Трив. (хв)</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>
<script>
const RAW = {json.dumps(records_json, ensure_ascii=False)};
const sectors = [...new Set(RAW.map(r => r.sector))];
const types = [...new Set(RAW.map(r => r.event_type))];
const dates = [...new Set(RAW.map(r => r.date))].sort();
const fSec = document.getElementById('fSector'), fTyp = document.getElementById('fType');
const fDS = document.getElementById('fDateStart'), fDE = document.getElementById('fDateEnd');
sectors.forEach(s => fSec.innerHTML += `<option value="${{s}}">${{s}}</option>`);
types.forEach(t => fTyp.innerHTML += `<option value="${{t}}">${{t}}</option>`);
dates.forEach(d => {{ fDS.innerHTML += `<option value="${{d}}">${{d}}</option>`; fDE.innerHTML += `<option value="${{d}}">${{d}}</option>`; }});
fDE.value = dates[dates.length - 1];

function updateDash() {{
  const s = fSec.value, t = fTyp.value, d1 = fDS.value, d2 = fDE.value;
  const rows = RAW.filter(r => (s === 'ALL' || r.sector === s) && (t === 'ALL' || r.event_type === t) && r.date >= d1 && r.date <= d2);
  const totCount = rows.reduce((a, b) => a + b.events_count, 0);
  const avgInt = rows.length ? (rows.reduce((a, b) => a + b.mean_intensity * b.events_count, 0) / Math.max(1, totCount)) : 0;
  const totDur = rows.reduce((a, b) => a + b.total_duration_min, 0);
  const actSec = new Set(rows.map(r => r.sector)).size;
  document.getElementById('kpiCount').textContent = totCount;
  document.getElementById('kpiInt').textContent = avgInt.toFixed(2);
  document.getElementById('kpiDur').textContent = totDur;
  document.getElementById('kpiSec').textContent = actSec;

  const byDate = {{}};
  dates.filter(d => d >= d1 && d <= d2).forEach(d => byDate[d] = 0);
  rows.forEach(r => byDate[r.date] = (byDate[r.date] || 0) + r.events_count);
  const dKeys = Object.keys(byDate).sort();
  const maxD = Math.max(1, ...Object.values(byDate));
  const svgL = document.getElementById('svgLine');
  svgL.innerHTML = '';
  let pts = [];
  dKeys.forEach((d, idx) => {{
    const x = 40 + (idx / Math.max(1, dKeys.length - 1)) * 450;
    const y = 185 - (byDate[d] / maxD) * 150;
    pts.push(`${{x}},${{y}}`);
    svgL.innerHTML += `<circle cx="${{x}}" cy="${{y}}" r="4" fill="#1F4E78"/><text x="${{x-10}}" y="${{y-8}}" font-size="10">${{byDate[d]}}</text>`;
    if (idx % 2 === 0) svgL.innerHTML += `<text x="${{x-14}}" y="208" font-size="10" fill="#4A5568">${{d.slice(5)}}</text>`;
  }});
  if (pts.length > 1) svgL.innerHTML = `<polyline fill="none" stroke="#1F4E78" stroke-width="2.5" points="${{pts.join(' ')}}"/>` + svgL.innerHTML;

  const bySec = {{}};
  rows.forEach(r => bySec[r.sector] = (bySec[r.sector] || 0) + r.events_count);
  const svgS = document.getElementById('svgSec');
  svgS.innerHTML = '';
  const maxS = Math.max(1, ...Object.values(bySec));
  Object.entries(bySec).forEach(([sec, val], i) => {{
    const y = 25 + i * 44;
    const w = (val / maxS) * 300;
    svgS.innerHTML += `<text x="10" y="${{y+18}}" font-size="11">${{sec}}</text><rect x="145" y="${{y}}" width="${{w}}" height="26" fill="#4B5320"/><text x="${{152+w}}" y="${{y+18}}" font-size="11" font-weight="bold">${{val}}</text>`;
  }});

  const byTyp = {{}};
  rows.forEach(r => byTyp[r.event_type] = (byTyp[r.event_type] || 0) + r.events_count);
  const svgT = document.getElementById('svgType');
  svgT.innerHTML = '';
  const cols = ["#A04030", "#1F4E78", "#B87333", "#4B5320"];
  Object.entries(byTyp).forEach(([tp, val], i) => {{
    const pct = totCount ? (val / totCount * 100) : 0;
    const y = 25 + i * 44;
    const w = (pct / 100) * 260;
    svgT.innerHTML += `<text x="10" y="${{y+18}}" font-size="11">${{tp}}</text><rect x="195" y="${{y}}" width="${{w}}" height="26" fill="${{cols[i%4]}}"/><text x="${{202+w}}" y="${{y+18}}" font-size="11" font-weight="bold">${{pct.toFixed(1)}}% (${{val}})</text>`;
  }});

  const tb = document.querySelector('#dataTable tbody');
  tb.innerHTML = rows.map(r => `<tr><td>${{r.date}}</td><td>${{r.sector}}</td><td>${{r.event_type}}</td><td>${{r.events_count}}</td><td>${{r.mean_intensity}}</td><td>${{r.total_duration_min}}</td></tr>`).join('');
}}
[fSec, fTyp, fDS, fDE].forEach(el => el.addEventListener('change', updateDash));
updateDash();
</script>
</body>
</html>"""
    with open(html_dash_path, "w", encoding="utf-8") as f:
        f.write(html_dash)

    return {
        "n_rows": len(dash_df),
        "total_events": int(dash_df["events_count"].sum()),
        "total_duration": int(dash_df["total_duration_min"].sum()),
    }


# ==============================================================================
# ВАРІАНТ 8: ПОРІВНЯННЯ БПЛА (BOXPLOT + VIOLIN PLOT НА ОКРЕМИХ ПІДГРАФІКАХ)
# ==============================================================================
def run_variant_8(csv_dir, png_dir):
    rng = np.random.default_rng(SEED + 8)
    uav_models = [
        ("БПЛА-Р1 «Лелека-Н»", "Оперативно-тактичний розвідник", 135, 14, 42, 4.5, 4.8, 0.45),
        ("БПЛА-Р2 «Фурія-Н»", "Тактичний розвідник", 92, 11, 28, 3.2, 3.6, 0.35),
        ("БПЛА-К3 «Сокіл-М»", "Коптер ближньої розвідки", 44, 6, 9.5, 1.4, 7.2, 0.75),
        ("БПЛА-У4 «Грім-У»", "Ударно-розвідувальний", 118, 16, 35, 5.0, 2.5, 0.30),
    ]
    n_flights_per_type = 45
    rows = []
    fid = 1
    for model_name, uav_class, m_dur, sd_dur, m_rng, sd_rng, m_err, sd_err in uav_models:
        for _ in range(n_flights_per_type):
            dur = round(float(max(18.0, rng.normal(m_dur, sd_dur))), 1)
            rad = round(float(max(2.5, rng.normal(m_rng, sd_rng))), 2)
            cep = round(float(max(0.8, rng.normal(m_err, sd_err))), 2)
            rows.append({
                "flight_id": f"FL-{fid:03d}",
                "uav_model": model_name,
                "uav_class": uav_class,
                "flight_duration_min": dur,
                "effective_range_km": rad,
                "target_cep_m": cep,
                "is_synthetic_training_data": True,
            })
            fid += 1

    df_raw = pd.DataFrame(rows)
    df_raw.loc[5, "flight_duration_min"] = -10.0
    df_raw.loc[78, "effective_range_km"] = np.nan
    df_raw.to_csv(os.path.join(csv_dir, "v8_uav_comparison_raw.csv"), index=False, encoding="utf-8")

    df_clean = df_raw.dropna().copy()
    df_clean = df_clean[df_clean["flight_duration_min"] > 0].reset_index(drop=True)
    df_clean.to_csv(os.path.join(csv_dir, "v8_uav_comparison_clean.csv"), index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8))
    metrics = [
        ("flight_duration_min", "Тривалість польоту, хв", "Підграфік А: Тривалість польоту (хв)"),
        ("effective_range_km", "Радіус дії каналу зв'язку, км", "Підграфік Б: Ефективний радіус дії (км)"),
        ("target_cep_m", "Кругове імовірне відхилення КІВ, м", "Підграфік В: Похибка визначення координат (м)"),
    ]
    model_order = [m[0] for m in uav_models]
    short_labels = ["БПЛА-Р1\n«Лелека-Н»", "БПЛА-Р2\n«Фурія-Н»", "БПЛА-К3\n«Сокіл-М»", "БПЛА-У4\n«Грім-У»"]
    box_colors = [PALETTE["nato_blue"], PALETTE["olive_dark"], PALETTE["earth_ochre"], PALETTE["earth_terracotta"]]

    stats_summary = {}
    for ax, (col, ylabel, title) in zip(axes, metrics):
        data_groups = [df_clean[df_clean["uav_model"] == m][col].values for m in model_order]
        parts = ax.violinplot(data_groups, positions=range(len(model_order)), showmeans=False, showmedians=False, widths=0.65)
        for pc, c in zip(parts["bodies"], box_colors):
            pc.set_facecolor(c)
            pc.set_edgecolor(c)
            pc.set_alpha(0.28)
        for partname in ("cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_edgecolor(PALETTE["slate_grey"])
                parts[partname].set_linewidth(0.8)

        bp = ax.boxplot(
            data_groups,
            positions=range(len(model_order)),
            widths=0.26,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markeredgecolor="white", markerfacecolor=PALETTE["earth_terracotta"], markersize=6),
            medianprops=dict(color="white", linewidth=2.0),
        )
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.82)

        for i, vals in enumerate(data_groups):
            med = np.median(vals)
            mn = np.mean(vals)
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            ax.text(
                i,
                np.max(vals) + (np.max(vals) - np.min(vals)) * 0.04,
                f"Med={med:.1f}\nMean={mn:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D8D6CE", alpha=0.9),
            )
            stats_summary.setdefault(model_order[i], {})[col] = {
                "mean": round(float(mn), 2),
                "median": round(float(med), 2),
                "std": round(float(np.std(vals, ddof=1)), 2),
                "iqr": round(float(iqr), 2),
            }

        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.16)

    legend_elements = [
        Line2D([0], [0], color="white", markerfacecolor=PALETTE["nato_blue"], marker="s", markersize=10, label="Міжквартильний розмах (IQR: Q25–Q75)"),
        Line2D([0], [0], color=PALETTE["slate_grey"], lw=2, label="Біла лінія всередині боксу — Медіана (Median)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PALETTE["earth_terracotta"], markersize=7, label="Ромб — Середнє арифметичне (Mean)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Варіант 8. Порівняння характеристик 4 типів БПЛА за повторними спостереженнями (n = 178 вильотів, Violin + Boxplot)",
        fontsize=12.2,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v8_uav_boxplot_violin.png"))
    plt.close(fig)

    return {
        "n_clean": len(df_clean),
        "stats_summary": stats_summary,
    }


# ==============================================================================
# ВАРІАНТ 9: ЛОГІСТИЧНИЙ МАРШРУТ (КАРТА МАРШРУТУ ТА ПРОФІЛЬ РИЗИКУ/ШВИДКОСТІ)
# ==============================================================================
def run_variant_9(csv_dir, png_dir, html_dir):
    waypoints = [
        (0, "КТ-01: Базовий склад", 49.20, 31.80, "Асфальт I кат.", "Немає", 62, 18, 1.4),
        (5, "КТ-02: Розвилка Північ", 49.23, 31.86, "Асфальт I кат.", "Немає", 65, 22, 1.6),
        (10, "КТ-03: Лісовий масив", 49.27, 31.92, "Асфальт II кат.", "Поодинокі вирви", 54, 38, 2.8),
        (15, "КТ-04: Понтонна переправа", 49.30, 31.98, "Ґрунтовий під'їзд", "Вузький міст / переправа", 24, 88, 7.6),
        (20, "КТ-05: Висота 142", 49.33, 32.05, "Щебеневе покриття", "Відкрита ділянка (БПЛА)", 38, 74, 6.8),
        (25, "КТ-06: Балка Довга", 49.36, 32.11, "Щебеневе покриття", "Крутий підйом 9%", 42, 62, 5.2),
        (30, "КТ-07: Проміжний хаб «Альфа»", 49.39, 32.18, "Асфальт II кат.", "Немає", 58, 35, 2.5),
        (35, "КТ-08: Залізничний переїзд", 49.41, 32.24, "Асфальт II кат.", "Зруйнований шляхопровід", 28, 82, 7.2),
        (40, "КТ-09: Лісосмуга Схід", 49.44, 32.31, "Ґрунтова дорога", "Розкисла ділянка", 31, 68, 5.9),
        (45, "КТ-10: Перехрестя Р-14", 49.47, 32.38, "Асфальт II кат.", "Зона періодичного обстрілу", 34, 91, 8.7),
        (50, "КТ-11: Яр Глибокий", 49.49, 32.44, "Щебеневе покриття", "Мінно-небезпечне узбіччя", 36, 79, 7.4),
        (55, "КТ-12: Рубіж маскування", 49.52, 32.50, "Асфальт II кат.", "Немає", 52, 48, 3.8),
        (60, "КТ-13: Розподільчий пункт", 49.54, 32.56, "Щебеневе покриття", "Вузький проїзд", 44, 55, 4.4),
        (65, "КТ-14: Передовий склад ППЗ", 49.56, 32.62, "Ґрунтовий капонір", "Немає", 40, 46, 3.9),
    ]
    cols = [
        "distance_km",
        "checkpoint_name",
        "latitude",
        "longitude",
        "road_surface",
        "obstacle_type",
        "avg_speed_kmh",
        "road_load_pct",
        "composite_risk_index",
    ]
    df_route = pd.DataFrame(waypoints, columns=cols)
    df_route["risk_category"] = pd.cut(
        df_route["composite_risk_index"],
        bins=[0, 3.5, 6.5, 10.0],
        labels=["Низький (<3.5)", "Середній (3.5–6.5)", "Високий (>6.5)"],
    )
    df_route["is_synthetic_training_data"] = True
    df_route.to_csv(os.path.join(csv_dir, "v9_logistics_route.csv"), index=False, encoding="utf-8")

    fig, (ax_map, ax_prof) = plt.subplots(
        1, 2, figsize=(14.6, 6.1), gridspec_kw={"width_ratios": [1.15, 1.35]}
    )

    ax_map.plot(
        df_route["longitude"],
        df_route["latitude"],
        color=PALETTE["slate_grey"],
        linewidth=2.8,
        linestyle="-",
        zorder=2,
        label="Траса навчального маршруту (65 км)",
    )
    map_offsets = {
        0: (8, -14),
        15: (10, -18),
        30: (-88, 12),
        35: (10, -18),
        45: (10, -16),
        50: (-92, 12),
        65: (-95, -6),
    }
    for _, r in df_route.iterrows():
        c = (
            PALETTE["earth_terracotta"]
            if r["composite_risk_index"] >= 6.5
            else (PALETTE["earth_ochre"] if r["composite_risk_index"] >= 3.5 else PALETTE["olive_dark"])
        )
        ax_map.scatter(r["longitude"], r["latitude"], s=110, color=c, edgecolors="white", linewidth=1.5, zorder=4)
        dkm = int(r["distance_km"])
        if dkm in map_offsets:
            ox, oy = map_offsets[dkm]
            ax_map.annotate(
                f"{r['checkpoint_name'].split(':')[0]} ({dkm} км)\nРизик: {r['composite_risk_index']}",
                (r["longitude"], r["latitude"]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=7.6,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=c, lw=0.8),
            )

    ax_map.set_xlim(31.74, 32.72)
    ax_map.set_ylim(49.16, 49.60)
    ax_map.set_title("Схема навчального логістичного маршруту (WGS-84)", fontsize=10.8, fontweight="bold")
    ax_map.set_xlabel("Довгота (Longitude, ° сх. д.)")
    ax_map.set_ylabel("Широта (Latitude, ° пн. ш.)")
    risk_Sample = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["olive_dark"], markersize=9, label="Низький ризик (<3.5 бала)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["earth_ochre"], markersize=9, label="Середній ризик (3.5–6.5 бала)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["earth_terracotta"], markersize=9, label="Високий ризик (>6.5 бала — перешкода)"),
    ]
    ax_map.legend(handles=risk_Sample, loc="lower right", fontsize=8)

    ax_prof.plot(
        df_route["distance_km"],
        df_route["avg_speed_kmh"],
        color=PALETTE["nato_blue"],
        marker="o",
        linewidth=2.2,
        label="Середня швидкість колони, км/год",
    )
    ax_prof.plot(
        df_route["distance_km"],
        df_route["road_load_pct"],
        color=PALETTE["olive_dark"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Завантаженість ділянки траси, %",
    )
    ax_prof.set_xlabel("Дистанція вздовж маршруту від базового складу, км")
    ax_prof.set_ylabel("Швидкість (км/год) / Завантаженість траси (%)")
    ax_prof.set_ylim(0, 112)

    ax_risk = ax_prof.twinx()
    ax_risk.plot(
        df_route["distance_km"],
        df_route["composite_risk_index"],
        color=PALETTE["earth_terracotta"],
        marker="^",
        linewidth=2.2,
        label="Інтегральний індекс ризику [0..10 балів]",
    )
    ax_risk.axhspan(6.5, 10.0, color=PALETTE["earth_terracotta"], alpha=0.10, label="Зона високого ризику (≥6.5 бала)")
    ax_risk.set_ylabel("Інтегральний індекс ризику, бали [0..10]", color=PALETTE["earth_terracotta"])
    ax_risk.set_ylim(0, 10.8)
    ax_risk.grid(False)

    prof_offsets = {
        15: (0, -24),
        20: (8, 18),
        35: (-10, -24),
        45: (-12, -26),
        50: (16, 16),
    }
    for _, r in df_route[df_route["composite_risk_index"] >= 6.8].iterrows():
        dkm = int(r["distance_km"])
        ox, oy = prof_offsets.get(dkm, (0, -22))
        ax_prof.annotate(
            f"{r['obstacle_type']}\n({dkm} км)",
            (r["distance_km"], r["avg_speed_kmh"]),
            textcoords="offset points",
            xytext=(ox, oy),
            ha="center",
            fontsize=7.3,
            fontweight="bold",
            color=PALETTE["earth_terracotta"],
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=PALETTE["earth_terracotta"], alpha=0.88),
            arrowprops=dict(arrowstyle="->", color=PALETTE["earth_terracotta"], lw=0.9),
        )

    lines1, labels1 = ax_prof.get_legend_handles_labels()
    lines2, labels2 = ax_risk.get_legend_handles_labels()
    ax_prof.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.8)
    ax_prof.set_title("Лінійний профіль швидкості, завантаженості та ризику вздовж маршруту", fontsize=10.8, fontweight="bold")

    fig.suptitle(
        "Варіант 9. Аналіз умовного логістичного маршруту та вузьких місць (Навчальні дані, 65 км)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(png_dir, "v9_logistics_route.png"))
    plt.close(fig)

    html_route_path = os.path.join(html_dir, "v9_logistics_route_map.html")
    route_json = df_route.drop(columns=["risk_category"]).to_dict(orient="records")
    with open(html_route_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="uk"><head><meta charset="utf-8"><title>Варіант 9 — Інтерактивна карта навчального логістичного маршруту</title>
<style>body{{font-family:'Segoe UI',Arial,sans-serif;background:#F7F6F2;padding:20px;color:#1C2321;}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px;margin-top:12px;}}th,td{{padding:6px 8px;border:1px solid #D8D6CE;}}th{{background:#1F4E78;color:#fff;}}</style>
</head><body>
<h2 style="color:#1F4E78;margin-top:0;">Варіант 9. Інтерактивна схема та таблиця контрольних точок навчального маршруту (65 км)</h2>
<p>Критерій ризику: <b>I_risk = 0.4·(Завантаженість/10) + 0.6·(Фактор перешкод та вразливості)</b> за шкалою 0–10 балів. Ділянки з I_risk ≥ 6.5 класифікуються як зони високого ризику.</p>
<table><thead><tr><th>Км</th><th>Контрольна точка</th><th>Координати (WGS-84)</th><th>Покриття</th><th>Наявна перешкода</th><th>Швидкість (км/год)</th><th>Завантаження (%)</th><th>Індекс ризику</th></tr></thead>
<tbody>
{"".join(f"<tr style='background:{'#FDECEC' if r['composite_risk_index']>=6.5 else '#fff'}'><td>{r['distance_km']}</td><td><b>{r['checkpoint_name']}</b></td><td>{r['latitude']}°N, {r['longitude']}°E</td><td>{r['road_surface']}</td><td>{r['obstacle_type']}</td><td>{r['avg_speed_kmh']}</td><td>{r['road_load_pct']}%</td><td><b>{r['composite_risk_index']}</b></td></tr>" for r in route_json)}
</tbody></table></body></html>""")

    return {
        "total_km": 65,
        "high_risk_segments": df_route[df_route["composite_risk_index"] >= 6.5][
            ["distance_km", "checkpoint_name", "obstacle_type", "avg_speed_kmh", "road_load_pct", "composite_risk_index"]
        ].to_dict(orient="records"),
        "mean_speed": round(float(df_route["avg_speed_kmh"].mean()), 1),
        "est_transit_time_min": round(float((5.0 / df_route["avg_speed_kmh"].iloc[1:]).sum() * 60.0), 1),
    }


# ==============================================================================
# ВАРІАНТ 10: СЦЕНАРІЇ ПРОГНОЗУВАННЯ (ОПТИМІСТИЧНИЙ, БАЗОВИЙ, ПЕСИМІСТИЧНИЙ)
# ==============================================================================
def run_variant_10(csv_dir, png_dir):
    rng = np.random.default_rng(SEED + 10)
    hist_dates = pd.date_range("2026-10-17", "2026-10-31", freq="D")
    fcast_dates = pd.date_range("2026-11-01", "2026-11-14", freq="D")

    hist_vals = [94.0]
    for _ in range(1, len(hist_dates)):
        hist_vals.append(round(hist_vals[-1] - 1.15 + float(rng.normal(0, 0.85)), 2))

    last_val = hist_vals[-1]
    rows = []
    for d, v in zip(hist_dates, hist_vals):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "period_type": "Фактичні ретроспективні дані (навчальні)",
            "actual_supply_pct": v,
            "baseline_scenario_pct": np.nan,
            "optimistic_scenario_pct": np.nan,
            "pessimistic_scenario_pct": np.nan,
            "normative_threshold_pct": 50.0,
            "is_synthetic_training_data": True,
        })

    for h_step, d in enumerate(fcast_dates, start=1):
        base_v = round(last_val - 1.20 * h_step, 2)
        opt_v = round(min(100.0, last_val + 0.70 * h_step), 2)
        pess_v = round(max(15.0, last_val - 3.70 * h_step), 2)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "period_type": "Сценарний прогноз (горизонт 14 діб)",
            "actual_supply_pct": np.nan,
            "baseline_scenario_pct": base_v,
            "optimistic_scenario_pct": opt_v,
            "pessimistic_scenario_pct": pess_v,
            "normative_threshold_pct": 50.0,
            "is_synthetic_training_data": True,
        })

    df_scen = pd.DataFrame(rows)
    df_scen.to_csv(os.path.join(csv_dir, "v10_forecasting_scenarios.csv"), index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    df_scen["date_dt"] = pd.to_datetime(df_scen["date"])
    hist_sub = df_scen[df_scen["actual_supply_pct"].notna()]
    fcast_sub = df_scen[df_scen["baseline_scenario_pct"].notna()]

    join_dates = pd.concat([hist_sub["date_dt"].iloc[-1:], fcast_sub["date_dt"]])
    join_base = [last_val] + fcast_sub["baseline_scenario_pct"].tolist()
    join_opt = [last_val] + fcast_sub["optimistic_scenario_pct"].tolist()
    join_pess = [last_val] + fcast_sub["pessimistic_scenario_pct"].tolist()

    ax.plot(
        hist_sub["date_dt"],
        hist_sub["actual_supply_pct"],
        color=PALETTE["text_dark"],
        marker="o",
        linewidth=2.4,
        label="Ретроспективні навчальні дані (17.10–31.10.2026)",
    )
    ax.plot(
        join_dates,
        join_opt,
        color=PALETTE["olive_dark"],
        marker="^",
        linestyle="--",
        linewidth=2.2,
        label="Оптимістичний сценарій (+0.7%/добу: стабільне підвезення та помірна витрата)",
    )
    ax.plot(
        join_dates,
        join_base,
        color=PALETTE["nato_blue"],
        marker="s",
        linestyle="-",
        linewidth=2.4,
        label="Базовий (інерційний) сценарій (-1.2%/добу: збереження поточного тренду)",
    )
    ax.plot(
        join_dates,
        join_pess,
        color=PALETTE["earth_terracotta"],
        marker="v",
        linestyle="-.",
        linewidth=2.2,
        label="Песимістичний сценарій (-3.7%/добу: затримки підвезення та пікова витрата)",
    )

    ax.fill_between(
        join_dates,
        join_pess,
        join_opt,
        color=PALETTE["earth_sand"],
        alpha=0.35,
        label="Сценарний діапазон невизначеності (між песимістичним та оптимістичним сценаріями)",
    )

    ax.axhline(
        50.0,
        color=PALETTE["earth_terracotta"],
        linestyle=":",
        linewidth=1.8,
        label="Критичний незнижуваний рубіж запасів (50% від норми)",
    )
    ax.axvline(hist_sub["date_dt"].iloc[-1], color=PALETTE["slate_grey"], linestyle="--", linewidth=1.2)
    ax.text(
        hist_sub["date_dt"].iloc[-1],
        97,
        " Точка початку прогнозу (31.10.2026)",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["slate_grey"],
    )

    d_end = join_dates.iloc[-1]
    for val, col, txt in [
        (join_opt[-1], PALETTE["olive_dark"], f"Оптиміст.: {join_opt[-1]:.1f}%"),
        (join_base[-1], PALETTE["nato_blue"], f"Базовий: {join_base[-1]:.1f}%"),
        (join_pess[-1], PALETTE["earth_terracotta"], f"Песиміст.: {join_pess[-1]:.1f}%"),
    ]:
        ax.annotate(
            txt,
            (d_end, val),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=8.5,
            fontweight="bold",
            color=col,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlim(hist_sub["date_dt"].iloc[0] - pd.Timedelta(days=1), d_end + pd.Timedelta(days=3.5))
    ax.set_ylim(15, 104)
    ax.set_xlabel("Дата (ретроспектива 15 діб та горизонт прогнозування 14 діб)")
    ax.set_ylabel("Рівень забезпеченості запасами МТЗ, % від нормативу")
    ax.set_title(
        "Варіант 10. Сценарне прогнозування рівня забезпеченості запасами МТЗ на горизонт 14 діб (Навчальні дані)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=8.2)

    plt.tight_layout()
    fig.savefig(os.path.join(png_dir, "v10_forecasting_scenarios.png"))
    plt.close(fig)

    cross_rows = fcast_sub[fcast_sub["pessimistic_scenario_pct"] <= 50.0]
    cross_date = cross_rows.iloc[0]["date"] if not cross_rows.empty else "не перетинає"

    return {
        "start_val_d0": last_val,
        "end_opt": join_opt[-1],
        "end_base": join_base[-1],
        "end_pess": join_pess[-1],
        "spread_d14": round(join_opt[-1] - join_pess[-1], 2),
        "pess_critical_cross_date": cross_date,
    }



# ==============================================================================
# ЄДИНЕ ВІКНО ПЕРЕГЛЯДУ ВСІХ 10 ВАРІАНТІВ (АВТОМАТИЧНО ВІДКРИВАЄТЬСЯ ПРИ RUN)
# ==============================================================================
def build_all_in_one_viewer(base_out, png_dir, html_dir, csv_dir, summary):
    """
    Формує єдиний автономний файл ALL_10_VARIANTS_VIEWER.html (усі графіки вбудовані
    у base64, інтерактивні карти та дашборд вбудовані у вкладки, таблиці та метрики)
    і автоматично відкриває його в одному вікні при натисканні Run / Run All.
    """
    def img_b64(fn):
        p = os.path.join(png_dir, fn)
        if not os.path.isfile(p):
            return ""
        with open(p, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

    cards = [
        ("Варіант 1. Карта подій (WGS-84) та фільтрування за часом", "v1_events_map.png", "v1_events_interactive_map.html", "v1_events_clean.csv",
         f"Очищено: {summary['variant_1']['n_clean']} подій (вилучено {summary['variant_1']['n_dups']} дублікати, {summary['variant_1']['n_invalid']} некоректних, виправлено {summary['variant_1']['n_swapped']} інвертовані координати). Середня інтенсивність: {summary['variant_1']['mean_intensity']} ± {summary['variant_1']['std_intensity']} бала. Пікова доба: {summary['variant_1']['peak_date']} ({summary['variant_1']['peak_date_count']} подій)."),
        ("Варіант 2. Динаміка атак за напрямками та порівняльна агрегація", "v2_attacks_dynamics.png", None, "v2_attacks_dynamics_clean.csv",
         f"Усього за 30 діб: {summary['variant_2']['total_attacks']} умовних атак. Лідер — Східний напрямок: Mean = 27.47 ± 5.04, Median = 27.0 атак/добу, максимум = 40 атак (2026-10-12)."),
        ("Варіант 3. Структура сил (Treemap особового складу та кільцева діаграма ОВТ)", "v3_force_structure.png", None, "v3_force_structure_clean.csv",
         f"Роздільний облік: особовий склад N = {summary['variant_3']['personnel_total']} осіб (механізовані 43.25%, забезпечення 22.50%, артилерія 18.75%, БПЛА/розвідка 15.50%); техніка M = {summary['variant_3']['equipment_total']} од. (ББМ 46.40%, артсистеми 27.20%, спецтехніка 14.00%, танки 12.40%)."),
        ("Варіант 4. Порівняння боєздатності (Радарна діаграма за 5 параметрами)", "v4_combat_capability_radar.png", None, "v4_combat_capability_clean.csv",
         f"Лідер за середнім балом нормалізованої шкали [0..100]: {summary['variant_4']['leader']} ({summary['variant_4']['leader_score']} бала). ОМБр-Б («Схід») = 89.14 бала, ОШБ-В («Південь») = 87.50 бала, БТрО-Г («Резерв») = 78.16 бала."),
        ("Варіант 5. Мережева взаємодія та показники центральності вузлів", "v5_network_interaction.png", None, "v5_network_centralities.csv",
         f"Орієнтований граф: {summary['variant_5']['n_nodes']} вузлів, {summary['variant_5']['n_edges']} канали. Найвища центральність посередництва (Betweenness): Ретранслятор-1 (C_B = 0.2556), КП Бр-1 та КП Бр-2 (C_B = 0.1574)."),
        ("Варіант 6. Кореляційний аналіз (Пірсон і Спірмен, циклічний час, змінна втрат)", "v6_correlation_heatmap.png", None, "v6_correlation_clean.csv",
         f"Вибірка n = {summary['variant_6']['n_clean']} епізодів. Зв'язок зі змінною умовних втрат: інтенсивність обстрілів r = 0.538 (ρ = 0.559), вильоти БПЛА r = 0.473, затримка реакції r = 0.393, інженерне обладнання позицій r = -0.499."),
        ("Варіант 7. Інтерактивний аналітичний дашборд (4 узгоджені панелі + фільтри)", "v7_dashboard_overview.png", "v7_interactive_dashboard.html", "v7_dashboard_dataset.csv",
         f"Зведений датасет дашборду: {summary['variant_7']['n_rows']} агрегованих груп ({summary['variant_7']['total_events']} подій, сумарна тривалість {summary['variant_7']['total_duration']} хв). Нижче вбудовано живий інтерактивний дашборд із фільтрами."),
        ("Варіант 8. Порівняння характеристик 4 типів БПЛА (Violin + Boxplot)", "v8_uav_boxplot_violin.png", None, "v8_uav_comparison_clean.csv",
         f"Повторні випробування n = {summary['variant_8']['n_clean']} вильотів. Тривалість польоту: лідер БПЛА-Р1 «Лелека-Н» (Median = 136.25 хв, радіус Median = 41.66 км). Точність (КІВ): лідер БПЛА-У4 «Грім-У» (Median = 2.37 м)."),
        ("Варіант 9. Логістичний маршрут (65 км) та лінійний профіль ризику", "v9_logistics_route.png", "v9_logistics_route_map.html", "v9_logistics_route.csv",
         f"Протяжність 65 км (14 КТ), середня швидкість {summary['variant_9']['mean_speed']} км/год, розрахунковий час {summary['variant_9']['est_transit_time_min']} хв. Виявлено 5 ділянок високого ризику (I_risk >= 6.5 бала): 15 км, 20 км, 35 км, 45 км, 50 км."),
        ("Варіант 10. Сценарії прогнозування запасів МТЗ на горизонт 14 діб", "v10_forecasting_scenarios.png", None, "v10_forecasting_scenarios.csv",
         f"Точка старту прогнозу (D0): {summary['variant_10']['start_val_d0']}%. На 14-ту добу: оптимістичний = {summary['variant_10']['end_opt']}%, базовий = {summary['variant_10']['end_base']}%, песимістичний = {summary['variant_10']['end_pess']}% (сценарний розмах {summary['variant_10']['spread_d14']} в. п.; перетин порогу 50% у песимістичному сценарії: {summary['variant_10']['pess_critical_cross_date']})."),
    ]

    nav_buttons = "".join(
        f'<button class="tab-btn" onclick="showTab({i})" id="btn-{i}">Вар. {i}</button>'
        for i in range(1, 11)
    )
    sections_html = ""
    for i, (title, png_fn, html_fn, csv_fn, desc) in enumerate(cards, start=1):
        b64_uri = img_b64(png_fn)
        iframe_block = ""
        if html_fn:
            rel_html = f"interactive_html/{html_fn}"
            iframe_block = f"""
            <div style="margin-top:16px;">
              <h4 style="margin:0 0 8px 0;color:#1F4E78;">Інтерактивний компонент ({html_fn}):</h4>
              <iframe src="{rel_html}" style="width:100%;height:620px;border:1px solid #CBD5E0;border-radius:6px;background:#fff;"></iframe>
            </div>"""
        sections_html += f"""
        <div class="variant-card" id="var-{i}">
          <div class="card-header">
            <h2>{title}</h2>
            <a class="csv-link" href="data_csv/{csv_fn}" download>Завантажити {csv_fn}</a>
          </div>
          <p class="summary-box"><b>Ключові результати розрахунків:</b> {desc}</p>
          <div class="img-wrap">
            <img src="{b64_uri}" alt="{title}">
          </div>
          {iframe_block}
        </div>"""

    full_html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Усі 10 варіантів в одному вікні — Методологія відображення даних в інтересах ІАЗ</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #F4F3EE; color: #1C2321; }}
  .top-bar {{ position: sticky; top: 0; z-index: 999; background: #1F4E78; color: #fff; padding: 12px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.18); display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
  .top-bar h1 {{ margin: 0; font-size: 17px; margin-right: auto; }}
  .tab-btn {{ background: rgba(255,255,255,0.15); color: #fff; border: 1px solid rgba(255,255,255,0.3); padding: 6px 11px; border-radius: 4px; cursor: pointer; font-size: 12.5px; font-weight: 600; }}
  .tab-btn:hover, .tab-btn.active {{ background: #fff; color: #1F4E78; }}
  .container {{ max-width: 1380px; margin: 20px auto; padding: 0 20px; }}
  .variant-card {{ background: #fff; border: 1px solid #D8D6CE; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1F4E78; padding-bottom: 10px; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }}
  .card-header h2 {{ margin: 0; font-size: 18px; color: #1F4E78; }}
  .csv-link {{ background: #4B5320; color: #fff; text-decoration: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .summary-box {{ background: #F7F6F2; border-left: 4px solid #4B5320; padding: 10px 14px; font-size: 13.5px; line-height: 1.5; margin: 0 0 14px 0; }}
  .img-wrap {{ text-align: center; }}
  .img-wrap img {{ max-width: 100%; height: auto; border: 1px solid #E2E8F0; border-radius: 4px; }}
</style>
</head>
<body>
<div class="top-bar">
  <h1>ІАЗ — Заняття 3: Єдине вікно перегляду (Варіанти 1–10, seed=42)</h1>
  <button class="tab-btn active" onclick="showTab(0)" id="btn-0">Показати всі 10 разом</button>
  {nav_buttons}
</div>
<div class="container">
  {sections_html}
</div>
<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + idx).classList.add('active');
  document.querySelectorAll('.variant-card').forEach((card, i) => {{
    card.style.display = (idx === 0 || idx === i + 1) ? 'block' : 'none';
  }});
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}
</script>
</body>
</html>"""

    viewer_path = os.path.join(base_out, "ALL_10_VARIANTS_VIEWER.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    return viewer_path


def main():
    base_out = os.path.abspath(
        os.environ.get(
            "IAZ_OUTPUT_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_iaz"),
        )
    )
    csv_dir, png_dir, html_dir = setup_dirs(base_out)

    summary = {}
    summary["variant_1"] = run_variant_1(csv_dir, png_dir, html_dir)
    summary["variant_2"] = run_variant_2(csv_dir, png_dir)
    summary["variant_3"] = run_variant_3(csv_dir, png_dir)
    summary["variant_4"] = run_variant_4(csv_dir, png_dir)
    summary["variant_5"] = run_variant_5(csv_dir, png_dir)
    summary["variant_6"] = run_variant_6(csv_dir, png_dir)
    summary["variant_7"] = run_variant_7(csv_dir, png_dir, html_dir)
    summary["variant_8"] = run_variant_8(csv_dir, png_dir)
    summary["variant_9"] = run_variant_9(csv_dir, png_dir, html_dir)
    summary["variant_10"] = run_variant_10(csv_dir, png_dir)

    summary_path = os.path.join(base_out, "calculated_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for fn in os.listdir(png_dir):
        src = os.path.join(png_dir, fn)
        dst = os.path.join(base_out, fn)
        if src != dst:
            with open(src, "rb") as fs, open(dst, "wb") as fd:
                fd.write(fs.read())

    for fn in os.listdir(html_dir):
        src = os.path.join(html_dir, fn)
        dst = os.path.join(base_out, fn)
        if src != dst:
            with open(src, "rb") as fs, open(dst, "wb") as fd:
                fd.write(fs.read())

    viewer_path = build_all_in_one_viewer(base_out, png_dir, html_dir, csv_dir, summary)
    print("Усі 10 варіантів успішно виконано!")
    print("Єдине вікно з усіма 10 варіантами створено:", viewer_path)

    # Автоматично відкриваємо єдине вікно у браузері при натисканні Run / Run All на ПК користувача
    if not os.environ.get("IAZ_OUTPUT_DIR"):
        try:
            if sys.platform.startswith("win"):
                os.startfile(viewer_path)
            else:
                webbrowser.open("file://" + os.path.abspath(viewer_path))
        except Exception as e:
            print("Відкрийте файл у браузері вручну:", viewer_path, e)


# %% [markdown]
# ### Запуск усіх 10 варіантів (підтримує як звичайний Run Python File, так і Run All у VS Code Interactive)
# %%
if __name__ == "__main__":
    main()