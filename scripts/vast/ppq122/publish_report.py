#!/usr/bin/env python3
"""Publish the wire-compression report into the efficient-inference site.

    python3 scripts/vast/ppq122/publish_report.py [--check]

Copies the report to compression/<slug>.html, retitles it, and inserts its card at
the index's insertion marker. Honours that repo's contract: self-contained page, no
external requests, no em dashes in published prose, newest card first, local links
validated. It does not commit and it does not push.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

SRC = Path("reports/pipeline-activation-wire-compression-report.html")
SITE = Path.home() / "Documents/efficient-inference"
SLUG = "initial-activation-boundary-quantization"
TITLE = "Initial Activation Boundary Quantization"
DESC = ("Measured quality and latency cost of quantizing only the activations sent across "
        "pipeline-parallel stage boundaries, for INT8, FP8, MXFP8, INT4, MXFP4 and NVFP4, "
        "at one, three and seven boundaries on a 35B and a 122B model.")
KICKER = "Experiment report"


def card(date, n_formats, n_depths):
    return f'''        <a class="report" href="{SLUG}.html">
          <span class="date">{date} &middot; {KICKER}</span>
          <h2>{TITLE}</h2>
          <p>Quantizing only the tensors that cross pipeline-parallel stage boundaries, leaving weights,
            KV cache and compute untouched. {n_formats} wire formats measured at {n_depths} pipeline depths on a
            35B mixture of experts model and a 122B model with FP8 weights, scored by teacher-forced
            WikiText NLL, GSM8K accuracy, a 1,024 step decode-position probe, and serving latency.
            Covers where 8-bit is free, why per-token INT4 collapses while microscaled 4-bit survives,
            and why the scale format rather than the block size separates MXFP4 from NVFP4.</p>
        </a>
'''


def check_contract(html, where):
    problems = []
    if "—" in html:
        problems.append(f"{where}: contains em dashes")
    for m in re.finditer(r'(?:src|href)="(https?://[^"]+)"', html):
        # A plain outbound hyperlink is fine; a resource load is not.
        seg = html[max(0, m.start() - 60):m.start()]
        if re.search(r"<(script|link|img|iframe|source|video|audio)\b[^>]*$", seg, re.I):
            problems.append(f"{where}: external resource load {m.group(1)[:60]}")
    if not html.lstrip().lower().startswith("<!doctype html>"):
        problems.append(f"{where}: missing doctype")
    for attr in ('lang=', 'charset=', 'name="viewport"', "<title>", 'name="description"'):
        if attr not in html[:2000]:
            problems.append(f"{where}: head missing {attr}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--date", default=None, help="card date, YYYY-MM-DD")
    a = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    if not SITE.exists():
        sys.exit(f"missing site repo {SITE}")
    html = SRC.read_text()

    problems = check_contract(html, "report")
    if problems:
        print("CONTRACT PROBLEMS:")
        for p in problems:
            print("  " + p)
        if not a.check:
            sys.exit(1)
    else:
        print("contract checks pass: doctype, head, no em dashes, no external resource loads")

    # Retitle for the site. The h1 stays as the editorial headline.
    out = html
    out = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", out, count=1)
    out = re.sub(r'<meta name="description" content=".*?">',
                 f'<meta name="description" content="{DESC}">', out, count=1)

    date = a.date or "2026-08-24"
    dest = SITE / "compression" / f"{SLUG}.html"
    idx = SITE / "compression/index.html"
    index_html = idx.read_text()
    marker = "<!-- compression:insert -->"
    if marker not in index_html:
        sys.exit("insertion marker not found in compression/index.html")
    already = f'href="{SLUG}.html"' in index_html

    print(f"\nplan:")
    print(f"  write  {dest}   ({len(out):,} bytes)")
    print(f"  title  {TITLE}")
    print(f"  card   {'REPLACE existing' if already else 'INSERT at marker (newest first)'}")

    if a.check:
        return

    dest.write_text(out)
    if already:
        index_html = re.sub(r'\s*<a class="report" href="%s\.html">.*?</a>\n' % re.escape(SLUG),
                            "\n", index_html, count=1, flags=re.S)
    n_formats, n_depths = "Six", "three"
    index_html = index_html.replace(marker, marker + "\n" + card(date, n_formats, n_depths).rstrip("\n"), 1)
    idx.write_text(index_html)

    # Validate local links in both files.
    bad = []
    for f in (dest, idx):
        for m in re.finditer(r'href="(?!https?:|#|mailto:)([^"]+)"', f.read_text()):
            target = (f.parent / m.group(1)).resolve()
            if not target.exists():
                bad.append(f"{f.name} -> {m.group(1)}")
    print(f"\nwrote {dest.name} and updated the index")
    print("local links: " + ("all resolve" if not bad else "BROKEN: " + ", ".join(bad)))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
