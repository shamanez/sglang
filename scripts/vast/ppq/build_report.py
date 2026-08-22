#!/usr/bin/env python3
"""Build the single-file HTML report for the PP activation wire-quant experiment.

Reads /workspace/ppq/results/<label>/{wikitext,gsm8k,probe,server_info}.json and
bench.jsonl, computes aggregates, renders matplotlib figures as inline SVG, and
writes one self-contained HTML file. Run on the box after run_all.sh completes:
    python3 build_report.py --out /workspace/ppq/pp_wire_quant_report.html
"""
import argparse
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = ["pp2_bf16", "pp2_int8", "pp2_int4", "pp4_bf16", "pp4_int8", "pp4_int4"]
BASELINE = "pp2_bf16"
COLORS = {
    "pp2_bf16": "#5C6470",
    "pp2_int8": "#0E6B5E",
    "pp2_int4": "#C25E00",
    "pp4_bf16": "#9AA3AC",
    "pp4_int8": "#43A08A",
    "pp4_int4": "#B3271E",
}
LABELS = {
    "pp2_bf16": "pp2 bf16 (baseline)",
    "pp2_int8": "pp2 int8 wire",
    "pp2_int4": "pp2 int4 wire",
    "pp4_bf16": "pp4 bf16",
    "pp4_int8": "pp4 int8 wire",
    "pp4_int4": "pp4 int4 wire",
}
LINESTYLES = {c: ("--" if c.startswith("pp4") else "-") for c in CONFIGS}
BUCKET = 16
PROBE_LEN = 256

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "svg.fonttype": "none",
    }
)


def load(results_dir):
    data = {}
    for c in CONFIGS:
        d = results_dir / c
        cfg = {}
        for name in ("wikitext", "gsm8k", "probe", "server_info"):
            p = d / f"{name}.json"
            cfg[name] = json.loads(p.read_text()) if p.exists() else None
        bench = d / "bench.jsonl"
        cfg["bench"] = None
        if bench.exists():
            lines = [json.loads(l) for l in bench.read_text().splitlines() if l.strip()]
            cfg["bench"] = lines[-1] if lines else None
        data[c] = cfg
    return data


def fig_to_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg") :]


def gsm8k_accuracy(g):
    """Numeric re-grade of the stored per-item results: the first run's
    string-equality grading scored '16.00' != '16' as wrong (4 points in
    every config)."""

    def match(pred, gold):
        try:
            return pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except (TypeError, ValueError):
            return False

    items = g["items"]
    return sum(match(it["pred"], it["gold"]) for it in items) / len(items)


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def probe_series(probe, key_fn):
    """Bucketed mean of key_fn(step, row) over positions k."""
    buckets = [[] for _ in range(PROBE_LEN // BUCKET)]
    for row in probe:
        for st in row["steps"]:
            b = st["k"] // BUCKET
            if b < len(buckets):
                v = key_fn(st, row)
                if v is not None:
                    buckets[b].append(v)
    xs, ys = [], []
    for b, vals in enumerate(buckets):
        if vals:
            xs.append(b * BUCKET + BUCKET / 2)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def top1_agree(st, row):
    if not st.get("top"):
        return None
    ref = row["ref_ids"][st["k"]]
    return 1.0 if st["top"][0][1] == ref else 0.0


def make_k3_fn(base_probe):
    """k3 estimator of KL(base || config) on the shared reference tokens —
    the same estimator sglang's kl_test_utils calibrates numerics changes
    with, computed from the per-position logprob pairs already collected."""
    base_nll = {(r["i"], st["k"]): st["nll"] for r in base_probe for st in r["steps"]}

    def k3(st, row):
        b = base_nll.get((row["i"], st["k"]))
        if b is None:
            return None
        logr = max(min(b - st["nll"], 20.0), -20.0)
        return (math.exp(logr) - 1) - logr

    return k3


def build(data, out_path, commit):
    figs = {}

    # ---- wikitext ----
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    means = {}
    for i, c in enumerate(CONFIGS):
        wt = data[c]["wikitext"]
        if not wt:
            continue
        per_win = [r["sum_nll"] / r["n_tokens"] for r in wt["windows"]]
        means[c] = wt["mean_nll"]
        ax.scatter([i] * len(per_win), per_win, s=12, alpha=0.45, color=COLORS[c])
        ax.scatter([i], [wt["mean_nll"]], s=140, marker="_", color=COLORS[c], linewidths=3)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], rotation=12)
    ax.set_ylabel("teacher-forced NLL (nats/token)")
    ax.set_title("WikiText-2: per-window NLL (dashes = config mean)")
    figs["wikitext"] = fig_to_svg(fig)

    # ---- gsm8k ----
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    gsm_n = None
    for i, c in enumerate(CONFIGS):
        g = data[c]["gsm8k"]
        if not g:
            continue
        p, n = gsm8k_accuracy(g), g["n"]
        gsm_n = n
        lo, hi = wilson_ci(p, n)
        ax.bar(i, p, color=COLORS[c], width=0.6)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="#222", capsize=4, fmt="none")
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], rotation=14, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exact-match accuracy (numeric grading)")
    ax.set_title(f"GSM8K (n={gsm_n}, greedy, 95% Wilson CI)")
    figs["gsm8k"] = fig_to_svg(fig)

    # ---- probe: NLL / agreement / KL vs position ----
    base_probe = data[BASELINE]["probe"]
    specs = [
        ("probe_nll", lambda st, row: st["nll"], "NLL of reference token (nats)",
         "Reference-trajectory NLL vs decode position (bucket = 16)", False),
        ("probe_agree", top1_agree, "top-1 agreement with reference",
         "Greedy agreement with the bf16 reference vs decode position", False),
    ]
    if base_probe:
        specs.append(
            ("probe_kl", make_k3_fn(base_probe), "KL(base ‖ config), k3 estimator (nats)",
             "KL to baseline on shared reference tokens vs decode position", True)
        )
    for name, key_fn, ylab, title, skip_baseline in specs:
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        for c in CONFIGS:
            probe = data[c]["probe"]
            if not probe:
                continue
            if skip_baseline and c == BASELINE:
                continue
            xs, ys = probe_series(probe, key_fn)
            ax.plot(xs, ys, label=LABELS[c], color=COLORS[c],
                    linestyle=LINESTYLES[c], marker="o", ms=3, lw=1.6)
        ax.set_xlabel("decode position k in the 256-token continuation")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        figs[name] = fig_to_svg(fig)

    # ---- bench ----
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    metrics = [
        ("median_ttft_ms", "median TTFT (ms)"),
        ("output_throughput", "output throughput (tok/s)"),
    ]
    for ax, (mk, ylab) in zip(axes, metrics):
        for i, c in enumerate(CONFIGS):
            b = data[c]["bench"]
            if not b or mk not in b:
                continue
            v = b[mk]
            ax.bar(i, v, color=COLORS[c], width=0.6)
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([c.replace("_", "\n") for c in CONFIGS], fontsize=8)
        ax.set_ylabel(ylab)
    fig.suptitle("bench_serving: 48 prompts, 1024 in / 256 out, concurrency 16")
    figs["bench"] = fig_to_svg(fig)

    # ---- aggregates for the tables ----
    agg = {}
    for c in CONFIGS:
        d = data[c]
        row = {"label": LABELS[c]}
        if d["wikitext"]:
            row["nll"] = d["wikitext"]["mean_nll"]
            row["ppl"] = math.exp(d["wikitext"]["mean_nll"])
            row["dnll_pct"] = (
                100.0 * (d["wikitext"]["mean_nll"] / data[BASELINE]["wikitext"]["mean_nll"] - 1)
                if data[BASELINE]["wikitext"]
                else None
            )
        if d["gsm8k"]:
            row["acc"] = gsm8k_accuracy(d["gsm8k"])
            row["acc_string_graded"] = d["gsm8k"]["accuracy"]
        if d["probe"]:
            xs, ys = probe_series(d["probe"], top1_agree)
            row["agree_first"] = ys[0] if ys else None
            row["agree_last"] = ys[-1] if ys else None
            n_all = [st["nll"] for r in d["probe"] for st in r["steps"]]
            row["probe_nll"] = sum(n_all) / len(n_all) if n_all else None
        if d["bench"]:
            row["ttft"] = d["bench"].get("median_ttft_ms")
            row["thr"] = d["bench"].get("output_throughput")
        if d["server_info"]:
            row["pp"] = d["server_info"].get("pp_size")
            row["tp"] = d["server_info"].get("tp_size")
        agg[c] = row

    (out_path.parent / "report_aggregates.json").write_text(json.dumps(agg, indent=1))
    html = render_html(figs, agg, commit)
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(html)} bytes)")
    print(json.dumps(agg, indent=1))


def render_html(figs, agg, commit):
    def table_rows():
        cols = [
            ("label", lambda v: v),
            ("pp", str), ("tp", str),
            ("nll", lambda v: f"{v:.5f}"),
            ("ppl", lambda v: f"{v:.3f}"),
            ("dnll_pct", lambda v: f"{v:+.3f}%"),
            ("acc", lambda v: f"{v:.2f}"),
            ("agree_first", lambda v: f"{v:.3f}"),
            ("agree_last", lambda v: f"{v:.3f}"),
            ("ttft", lambda v: f"{v:.0f}"),
            ("thr", lambda v: f"{v:.0f}"),
        ]
        rows = ""
        for c in CONFIGS:
            r = agg.get(c, {})
            tds = "".join(
                f"<td>{fmt(r[k]) if r.get(k) is not None else '—'}</td>"
                for k, fmt in cols
            )
            rows += f"<tr>{tds}</tr>"
        return rows

    sections = "".join(
        f'<figure>{figs[k]}</figure>' for k in
        ("wikitext", "gsm8k", "probe_nll", "probe_agree", "probe_kl", "bench")
        if k in figs
    )
    # Narrative placeholders are replaced by the author before publishing.
    return f"""<meta charset="utf-8">
<title>PP Wire Quantization</title>
<style>
:root {{ --bg:#F7F8F5; --ink:#20242A; --muted:#5C6470; --line:#DDE2DA; --accent:#0E6B5E; --alert:#B3271E; --surface:#fff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14171B; --ink:#E7EAE4; --muted:#9AA3AC; --line:#2C3238; --accent:#4FC3AF; --alert:#FF8073; --surface:#1C2025; }} }}
:root[data-theme="dark"] {{ --bg:#14171B; --ink:#E7EAE4; --muted:#9AA3AC; --line:#2C3238; --accent:#4FC3AF; --alert:#FF8073; --surface:#1C2025; }}
body {{ background:var(--bg); color:var(--ink); font-family:system-ui,sans-serif; line-height:1.55; margin:0 auto; max-width:900px; padding:40px 24px 90px; }}
h1 {{ font-size:2rem; margin-bottom:4px; }} h2 {{ border-top:2px solid var(--ink); padding-top:12px; margin-top:44px; }}
.mono {{ font-family:ui-monospace,monospace; font-size:.85em; }}
figure {{ margin:18px 0; background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto; }}
figure svg {{ max-width:100%; height:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; }} td,th {{ border:1px solid var(--line); padding:6px 9px; text-align:left; }}
thead th {{ background:var(--surface); font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
.note {{ color:var(--muted); font-size:.85rem; }}
</style>
<h1>8-bit Activations on the Pipeline Wire</h1>
<p class="note">Qwen/Qwen3.6-35B-A3B · 4× RTX 5090 · sglang dev @ <span class="mono">{commit}</span> · run 2026-08-21</p>
<!--NARRATIVE_INTRO-->
<h2>Wire format</h2>
<div style="overflow-x:auto"><table><thead><tr>
<th>wire</th><th>payload per token per tensor (H=2048)</th><th>+ scale</th><th>per boundary (2 tensors)</th><th>compression</th>
</tr></thead><tbody>
<tr><td>bf16 (stock)</td><td>4,096 B</td><td>—</td><td>8,192 B</td><td>1.00×</td></tr>
<tr><td>int8 + f32/token scale</td><td>2,048 B</td><td>4 B</td><td>4,104 B</td><td>2.00×</td></tr>
<tr><td>int4 (packed) + f32/token scale</td><td>1,024 B</td><td>4 B</td><td>2,056 B</td><td>3.98×</td></tr>
</tbody></table></div>
<h2>Headline table</h2>
<div style="overflow-x:auto"><table><thead><tr>
<th>config</th><th>pp</th><th>tp</th><th>wikitext NLL</th><th>PPL</th><th>ΔNLL vs base</th><th>GSM8K acc</th><th>agree k∈[0,16)</th><th>agree k∈[240,256)</th><th>TTFT ms</th><th>out tok/s</th>
</tr></thead><tbody>{table_rows()}</tbody></table></div>
<h2>Plots</h2>
{sections}
<!--NARRATIVE_BODY-->
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/workspace/ppq/results")
    ap.add_argument("--out", default="/workspace/ppq/pp_wire_quant_report.html")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    build(load(Path(args.results)), Path(args.out), args.commit)
