#!/usr/bin/env python3
"""Fill the FP8 and MXFP8 gaps in the narrative report from the archived results.

    python3 scripts/vast/ppq122/update_report.py [--data reports/data]
                                                 [--report reports/pipeline-...html]
                                                 [--check]

Every number is recomputed from reports/data rather than typed in, and every edit
is an exact-string replacement that fails loudly if the anchor is not found, so a
report whose wording drifted cannot be silently half-updated. --check computes and
prints the numbers without touching the file.

The formulas were verified against values already in the report: NLL and latency
deltas reproduce the published INT8 cells to the last digit, and the 8-segment
agreement series reproduces all four published long-probe series exactly.
"""
import argparse
import json
import sys
from pathlib import Path

WIRE = "fp8"


def read_json(p):
    return json.loads(Path(p).read_text())


def bench(d):
    lines = [l for l in (Path(d) / "bench.jsonl").read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def segments(path, segs=8):
    """Top-1 agreement per decode segment, as percentages — the report's series format."""
    items = read_json(path)
    n = len(items[0]["steps"])
    agree = [0.0] * segs
    cnt = [0] * segs
    for it in items:
        refs = it["ref_ids"]
        for s in it["steps"]:
            i = min(s["k"] * segs // n, segs - 1)
            agree[i] += 1.0 if s["top"][0][1] == refs[s["k"]] else 0.0
            cnt[i] += 1
    return [round(a / c * 100, 2) for a, c in zip(agree, cnt)]


def collect(data: Path):
    grid, ppq8 = data / "main_grid", data / "ppq8"
    out = {}

    # Quality at 7 boundaries, against that topology's own bf16 control.
    base_nll = read_json(ppq8 / "results/pp8_bf16/wikitext.json")["mean_nll"]
    fp8_nll = read_json(ppq8 / f"results/pp8_{WIRE}/wikitext.json")["mean_nll"]
    out["nll7_raw"] = fp8_nll
    out["nll7"] = round((fp8_nll / base_nll - 1) * 100, 4)
    out["gsm7"] = round(read_json(ppq8 / f"results/pp8_{WIRE}/gsm8k.json")["accuracy"] * 100)

    # Latency at all three depths. 1 and 3 were measured long ago and never plotted.
    dirs = {1: grid / "pp2_", 3: grid / "pp4_", 7: ppq8 / "results/pp8_"}
    for depth, stem in dirs.items():
        b = bench(str(stem) + WIRE)
        c = bench(str(stem) + "bf16")
        out[f"ttft{depth}"] = round((b["median_ttft_ms"] / c["median_ttft_ms"] - 1) * 100, 2)
        out[f"itl{depth}"] = round((b["median_itl_ms"] / c["median_itl_ms"] - 1) * 100, 1)
        out[f"raw{depth}"] = f"{b['median_ttft_ms']:.0f} · {b['median_itl_ms']:.2f}"

    # 1024-step decode probe at 7 boundaries.
    lp = ppq8 / "results_longprobe"
    out["seg_fp8"] = segments(lp / f"lp2_pp8_{WIRE}/probe.json")
    out["seg_mxfp8"] = segments(lp / "lp2_pp8_mxfp8/probe.json")

    # Control: did this box reproduce the anchor the deltas are computed against?
    ctl = ppq8 / "results/ctl_pp8_bf16/wikitext.json"
    out["control_nll"] = read_json(ctl)["mean_nll"] if ctl.exists() else None
    out["control_anchor"] = base_nll
    return out


def series_literal(values):
    return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"


def edits(v):
    """(anchor, replacement, why) — anchors are exact and must each match once."""
    E = []

    # --- chart data -------------------------------------------------------
    E.append((
        '{ key: "fp8", label: "FP8", values: [{ x: 1, y: -0.0048 }, { x: 3, y: 0.0561 }] },',
        f'{{ key: "fp8", label: "FP8", values: [{{ x: 1, y: -0.0048 }}, {{ x: 3, y: 0.0561 }}, {{ x: 7, y: {v["nll7"]} }}] }},',
        "depthSafeConfig: add the 7-boundary NLL point",
    ))
    E.append((
        """        annotations: [
          { key: "fp8", x: 1, y: 1.75, anchor: "start", text: "FP8 not run at 7 boundaries" }
        ]
""",
        "",
        "depthSafeConfig: drop the 'not run' annotation",
    ))
    E.append((
        '{ key: "fp8", label: "FP8", values: [{ x: 1, y: 96 }, { x: 3, y: 94 }] },',
        f'{{ key: "fp8", label: "FP8", values: [{{ x: 1, y: 96 }}, {{ x: 3, y: 94 }}, {{ x: 7, y: {v["gsm7"]} }}] }},',
        "gsm35EightBitConfig: add the 7-boundary accuracy",
    ))
    E.append((
        '          { key: "nvfp4", label: "NVFP4", values: [97.27, 98.37, 98.37, 98.63, 97.92, 97.66, 97.66, 98.31].map((y, i) => ({ x: i + 1, y })) }\n',
        '          { key: "nvfp4", label: "NVFP4", values: [97.27, 98.37, 98.37, 98.63, 97.92, 97.66, 97.66, 98.31].map((y, i) => ({ x: i + 1, y })) },\n'
        f'          {{ key: "fp8", label: "FP8", values: {series_literal(v["seg_fp8"])}.map((y, i) => ({{ x: i + 1, y }})) }},\n'
        f'          {{ key: "mxfp8", label: "MXFP8", values: {series_literal(v["seg_mxfp8"])}.map((y, i) => ({{ x: i + 1, y }})) }}\n',
        "timeSafeConfig: add the fp8 and mxfp8 decode-horizon series",
    ))
    E.append((
        '          { key: "nvfp4", label: "NVFP4", values: [{ x: 1, y: -10.12 }, { x: 3, y: -16.67 }, { x: 7, y: -1.71 }] }\n',
        '          { key: "nvfp4", label: "NVFP4", values: [{ x: 1, y: -10.12 }, { x: 3, y: -16.67 }, { x: 7, y: -1.71 }] },\n'
        f'          {{ key: "fp8", label: "FP8", values: [{{ x: 1, y: {v["ttft1"]} }}, {{ x: 3, y: {v["ttft3"]} }}, {{ x: 7, y: {v["ttft7"]} }}] }}\n',
        "ttftConfig: add fp8 at all three depths (1 and 3 were measured, never plotted)",
    ))

    # --- tables -----------------------------------------------------------
    E.append((
        '<tr><th>MXFP8</th><td class="num">+9.0%</td><td class="num">+6.0%</td><td class="num">+9.7%</td><td>Microscale handling adds launch work.</td></tr>',
        '<tr><th>MXFP8</th><td class="num">+9.0%</td><td class="num">+6.0%</td><td class="num">+9.7%</td><td>Microscale handling adds launch work.</td></tr>\n'
        f'                <tr><th>FP8</th><td class="num">{v["itl1"]:+.1f}%</td><td class="num">{v["itl3"]:+.1f}%</td>'
        f'<td class="num">{v["itl7"]:+.1f}%</td><td>Per-token float codec, close to INT8.</td></tr>',
        "ITL table: add the FP8 row",
    ))
    E.append((
        '<tr><th>MXFP8</th><td class="num">317 · 14.61</td><td class="num">246 · 18.53</td><td class="num">188 · 24.31</td></tr>',
        '<tr><th>MXFP8</th><td class="num">317 · 14.61</td><td class="num">246 · 18.53</td><td class="num">188 · 24.31</td></tr>\n'
        f'                  <tr><th>FP8</th><td class="num">{v["raw1"]}</td><td class="num">{v["raw3"]}</td>'
        f'<td class="num">{v["raw7"]}</td></tr>',
        "raw medians table: add the FP8 row",
    ))

    # --- prose that asserts the gap --------------------------------------
    E.append((
        "E4M3 floating-point values with one FP32 scale per token. It was tested at one and three boundaries.",
        "E4M3 floating-point values with one FP32 scale per token. It was tested at one, three, and seven boundaries.",
        "format map: FP8 depth coverage",
    ))
    E.append((
        "Quality-safe in the four-GPU grid",
        "Quality-safe through seven boundaries",
        "format map table: FP8 observed role",
    ))
    E.append((
        "The leader changes at shallower depths, FP8 was not run at seven boundaries, and every 8-bit format lands within a few GSM8K questions of the others.",
        f"The leader changes at shallower depths, FP8 lands at {v['nll7']:+.2f}% once measured at seven boundaries, and every 8-bit format lands within a few GSM8K questions of the others.",
        "prose: the 8-bit comparison sentence",
    ))
    E.append((
        '<span class="marker-symbol" style="--marker-color: var(--blue)">■</span>FP8 (1 and 3 boundaries only)',
        '<span class="marker-symbol" style="--marker-color: var(--blue)">■</span>FP8',
        "depth chart legend: drop the caveat",
    ))
    E.append((
        "FP8 was measured only at one and three boundaries, at −0.00% and +0.06%. Its line ends at three because no seven-boundary run exists.",
        f"FP8 measures −0.00%, +0.06%, and {v['nll7']:+.2f}% at one, three, and seven boundaries.",
        "depth figcaption",
    ))
    E.append((
        "BF16, INT8, FP8, and MXFP8 on the shared scale. FP8 was not run at seven boundaries, so its slot there is empty.",
        "BF16, INT8, FP8, and MXFP8 on the shared scale.",
        "GSM8K panel blurb",
    ))
    E.append((
        "FP8 scores 96% and 94% at one and three boundaries and was not run at seven.",
        f"FP8 scores 96%, 94%, and {v['gsm7']}% at one, three, and seven boundaries.",
        "GSM8K figcaption",
    ))
    E.append((
        '<span class="legend-item"><span class="marker-symbol" style="--marker-color: var(--magenta)">\u2b1f</span>NVFP4</span>\n              </div>\n            </div>\n            <div class="chart-panel">\n              <h4>INT4 failure level</h4>',
        '<span class="legend-item"><span class="marker-symbol" style="--marker-color: var(--magenta)">\u2b1f</span>NVFP4</span>\n'
        '                <span class="legend-item"><span class="marker-symbol" style="--marker-color: var(--blue)">\u25a0</span>FP8</span>\n'
        '                <span class="legend-item"><span class="marker-symbol" style="--marker-color: var(--purple)">\u25b2</span>MXFP8</span>\n'
        '              </div>\n            </div>\n            <div class="chart-panel">\n              <h4>INT4 failure level</h4>',
        "decode-horizon legend: add FP8 and MXFP8",
    ))
    E.append((
        "BF16, INT8, MXFP4, and NVFP4 show no downward trend across eight segments.",
        "BF16, INT8, FP8, MXFP8, MXFP4, and NVFP4 show no downward trend across eight segments.",
        "decode-horizon figcaption: name the two added series",
    ))
    E.append((
        """        // One slot per series for the whole chart, not per group, so every bar in the
        // figure is the same width and a format keeps the same position in every
        // group. A format we never ran at a given depth leaves its slot empty, which
        // is how the missing FP8 seven-boundary run stays visible as a gap.""",
        """        // One slot per series for the whole chart, not per group, so every bar in the
        // figure is the same width and a format keeps the same position in every
        // group. A format we never ran at a given depth leaves its slot empty, so a
        // gap in the matrix stays visible rather than closing up.""",
        "renderer comment: the FP8 gap it describes is now filled",
    ))
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reports/data")
    ap.add_argument("--report", default="reports/pipeline-activation-wire-compression-report.html")
    ap.add_argument("--check", action="store_true", help="compute and print, do not edit")
    a = ap.parse_args()

    v = collect(Path(a.data))
    print("Recomputed from the archive:")
    print(f"  fp8 @ 7 boundaries   NLL {v['nll7_raw']!r}  ({v['nll7']:+.4f}% vs bf16)   GSM8K {v['gsm7']}%")
    for d in (1, 3, 7):
        print(f"  fp8 @ {d} boundar{'y' if d == 1 else 'ies'}    TTFT {v[f'ttft{d}']:+.2f}%   ITL {v[f'itl{d}']:+.1f}%   raw {v[f'raw{d}']}")
    print(f"  fp8   1024-probe     {v['seg_fp8']}")
    print(f"  mxfp8 1024-probe     {v['seg_mxfp8']}")
    if v["control_nll"] is not None:
        same = v["control_nll"] == v["control_anchor"]
        print(f"  control              {v['control_nll']!r} vs anchor {v['control_anchor']!r} -> "
              + ("BIT-IDENTICAL" if same else "DRIFTED, deltas above are suspect"))

    if a.check:
        return

    p = Path(a.report)
    html = p.read_text()
    applied = []
    for anchor, repl, why in edits(v):
        n = html.count(anchor)
        if n != 1:
            sys.exit(f"ABORT: anchor for '{why}' matched {n} times, expected 1.\n  {anchor[:110]}")
        html = html.replace(anchor, repl)
        applied.append(why)
    p.write_text(html)
    print(f"\nApplied {len(applied)} edits to {p}:")
    for w in applied:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
