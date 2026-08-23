#!/usr/bin/env python3
"""Render the 8-GPU stages (pp8 depth extension + 122B replication) as an HTML
section with inline SVG figures, for injection into the main report's narrative.

Runs locally against the archived result tree, not on the box:
    python3 build_8gpu_section.py --data reports/data --out section.html

Expects, under --data:
    ppq8/results/pp8_*                     35B, 7 boundaries
    ppq8/results_longprobe/lp2_pp8_*       35B, 1024-step probe at 7 boundaries
    ppq8/results_122b_pp8/q122_pp8_*       122B-FP8, 7 boundaries
    ppq8/results_122b_pp4/q122_pp4_*       122B-FP8, 3 boundaries
plus the 4-GPU campaign's main_grid/ for the 1- and 3-boundary 35B points.
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIRES = ["bf16", "int8", "mxfp8", "int4", "mxfp4", "nvfp4"]
COLORS = {
    "bf16": "#5C6470",
    "int8": "#0E6B5E",
    "mxfp8": "#5B3E96",
    "int4": "#B3271E",
    "mxfp4": "#C4A000",
    "nvfp4": "#A0246E",
}
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "svg.fonttype": "none",
    }
)


def fig_to_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg") :]


def numeric_ok(item):
    try:
        return abs(float(item["pred"]) - float(item["gold"])) < 1e-6
    except (TypeError, ValueError):
        return bool(item.get("correct"))


def read_cfg(d: Path):
    """Return {nll, acc, ttft, itl, thr} for one config dir, None fields if absent."""
    out = {k: None for k in ("nll", "acc", "ttft", "itl", "thr")}
    if not d.is_dir():
        return out
    wiki = d / "wikitext.json"
    if wiki.exists():
        out["nll"] = json.loads(wiki.read_text())["mean_nll"]
    gsm = d / "gsm8k.json"
    if gsm.exists():
        items = json.loads(gsm.read_text()).get("items", [])
        if items:
            out["acc"] = sum(map(numeric_ok, items)) / len(items)
    bench = d / "bench.jsonl"
    if bench.exists():
        lines = [l for l in bench.read_text().splitlines() if l.strip()]
        if lines:
            b = json.loads(lines[-1])
            out["ttft"] = b.get("median_ttft_ms")
            out["itl"] = b.get("median_itl_ms")
            out["thr"] = b.get("total_throughput")
    return out


def probe_segments(path: Path, segs: int):
    """Per-segment top-1 agreement against the frozen references."""
    if not path.exists():
        return None
    items = json.loads(path.read_text())
    items = items if isinstance(items, list) else items.get("items", items)
    if not items:
        return None
    n = len(items[0]["steps"])
    agree = [0.0] * segs
    count = [0] * segs
    for item in items:
        refs = item["ref_ids"]
        for s in item["steps"]:
            i = min(s["k"] * segs // n, segs - 1)
            agree[i] += 1.0 if s["top"][0][1] == refs[s["k"]] else 0.0
            count[i] += 1
    return [a / c for a, c in zip(agree, count) if c]


def pct(delta_from, value):
    if delta_from in (None, 0) or value is None:
        return None
    return (value / delta_from - 1) * 100


def collect(data: Path):
    """35B depth series (1/3/7 boundaries) + both 122B topologies."""
    grid = data / "main_grid"
    ppq8 = data / "ppq8"
    series = {"35b": {}, "122b": {}}
    for wire in WIRES:
        suffix = "bf16" if wire == "bf16" else wire
        series["35b"][wire] = {
            1: read_cfg(grid / f"pp2_{suffix}"),
            3: read_cfg(grid / f"pp4_{suffix}"),
            7: read_cfg(ppq8 / "results" / f"pp8_{suffix}"),
        }
    for wire in ["bf16", "int8", "int4", "mxfp4", "nvfp4"]:
        series["122b"][wire] = {
            3: read_cfg(ppq8 / "results_122b_pp4" / f"q122_pp4_{wire}"),
            7: read_cfg(ppq8 / "results_122b_pp8" / f"q122_pp8_{wire}"),
        }
    return series


def fig_depth(series, model, boundaries, title, ylabel, key, logy):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    base = {b: series[model]["bf16"][b][key] for b in boundaries}
    for wire, per_depth in series[model].items():
        if wire == "bf16":
            continue
        xs, ys = [], []
        for b in boundaries:
            v, ref = per_depth[b][key], base.get(b)
            if v is None or not ref:
                continue
            xs.append(b)
            ys.append(pct(ref, v))
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[wire], label=wire, linewidth=1.8)
    ax.axhline(0, color=COLORS["bf16"], linestyle=":", linewidth=1.2, label="bf16 control")
    if logy:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(boundaries)
    ax.set_xlabel("pipeline boundaries crossed per forward pass")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return fig_to_svg(fig)


def fig_acc(series, model, boundaries, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for wire, per_depth in series[model].items():
        xs = [b for b in boundaries if per_depth[b]["acc"] is not None]
        ys = [per_depth[b]["acc"] * 100 for b in xs]
        if xs:
            style = ":" if wire == "bf16" else "-"
            ax.plot(xs, ys, "o" + style, color=COLORS[wire], label=wire, linewidth=1.8)
    ax.set_xticks(boundaries)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("pipeline boundaries crossed per forward pass")
    ax.set_ylabel("GSM8K exact match (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return fig_to_svg(fig)


def fig_longprobe(data: Path):
    ppq8 = data / "ppq8" / "results_longprobe"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    any_series = False
    for wire in ["bf16", "int8", "nvfp4", "mxfp4", "int4"]:
        seg = probe_segments(ppq8 / f"lp2_pp8_{wire}" / "probe.json", 8)
        if not seg:
            continue
        any_series = True
        xs = [(i + 0.5) * 128 for i in range(len(seg))]
        ax.plot(xs, seg, "o-", color=COLORS[wire], label=wire, linewidth=1.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("decode position (1024-step horizon, 128-step segments)")
    ax.set_ylabel("top-1 agreement vs frozen reference")
    ax.set_title("35B, 7 boundaries: no temporal accumulation at 4x horizon")
    ax.legend(fontsize=8, ncol=2)
    return fig_to_svg(fig) if any_series else None


def row(label, cells):
    tds = "".join(f"<td>{c}</td>" for c in cells)
    return f"<tr><td>{label}</td>{tds}</tr>"


def fmt_nll(cfg, base):
    if cfg["nll"] is None:
        return "—"
    d = pct(base, cfg["nll"])
    return f"{cfg['nll']:.5f}" + (f" ({d:+.2f}%)" if d is not None else "")


def fmt_acc(cfg):
    return "—" if cfg["acc"] is None else f"{cfg['acc'] * 100:.0f}%"


def build(data: Path, out: Path):
    s = collect(data)
    parts = ["<h2>Scaling up: 7 boundaries and a 3.5x larger model</h2>"]
    parts.append(
        "<p>Two axes the four-GPU campaign could not reach: <strong>depth</strong> "
        "(pp8/tp1 crosses 7 boundaries per forward pass, against 1 and 3 before) and "
        "<strong>model size</strong> (<span class='mono'>Qwen3.5-122B-A10B-FP8</span>, "
        "hidden 3,072 &rarr; 12,288 B/token/boundary, 1.5x the 35B payload). Same codecs, "
        "same harness, same frozen references for the 35B rows so all three depths are "
        "directly comparable.</p>"
    )

    # 35B depth table
    b35 = [1, 3, 7]
    head = "".join(f"<th>{b} boundary{'s' if b > 1 else ''}</th>" for b in b35)
    rows = []
    for wire in WIRES:
        cells = []
        for b in b35:
            cfg = s["35b"][wire][b]
            base = s["35b"]["bf16"][b]["nll"]
            cells.append(f"{fmt_nll(cfg, base)}<br><span class='note'>{fmt_acc(cfg)}</span>")
        rows.append(row(wire, cells))
    parts.append(
        "<h3>35B: NLL (delta vs that topology's own bf16 control) and GSM8K by depth</h3>"
        f"<div style='overflow-x:auto'><table><thead><tr><th>wire</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    parts.append(f"<figure>{fig_depth(s, '35b', b35, '35B: quality cost vs pipeline depth', 'NLL delta vs bf16 control (%, symlog)', 'nll', True)}</figure>")
    parts.append(f"<figure>{fig_acc(s, '35b', b35, '35B: task accuracy vs pipeline depth')}</figure>")

    # 122B table
    b122 = [3, 7]
    head2 = "".join(f"<th>{b} boundaries</th>" for b in b122)
    rows2 = []
    for wire in ["bf16", "int8", "int4", "mxfp4", "nvfp4"]:
        cells = []
        for b in b122:
            cfg = s["122b"][wire][b]
            base = s["122b"]["bf16"][b]["nll"]
            cells.append(f"{fmt_nll(cfg, base)}<br><span class='note'>{fmt_acc(cfg)}</span>")
        rows2.append(row(wire, cells))
    parts.append(
        "<h3>122B-FP8: the same comparison on a larger model</h3>"
        f"<div style='overflow-x:auto'><table><thead><tr><th>wire</th>{head2}</tr></thead>"
        f"<tbody>{''.join(rows2)}</tbody></table></div>"
    )
    if any(s["122b"][w][3]["nll"] for w in s["122b"]):
        parts.append(f"<figure>{fig_depth(s, '122b', b122, '122B-FP8: quality cost vs pipeline depth', 'NLL delta vs bf16 control (%, symlog)', 'nll', True)}</figure>")
        parts.append(f"<figure>{fig_acc(s, '122b', b122, '122B-FP8: task accuracy vs pipeline depth')}</figure>")

    lp = fig_longprobe(data)
    if lp:
        parts.append("<h3>1024 decode steps at 7 boundaries</h3>")
        parts.append(f"<figure>{lp}</figure>")

    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reports/data")
    ap.add_argument("--out", default="reports/data/section_8gpu.html")
    a = ap.parse_args()
    build(Path(a.data), Path(a.out))
