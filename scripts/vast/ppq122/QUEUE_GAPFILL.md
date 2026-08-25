# Gap-fill run queue — 8x RTX 5090

Scoped to one question: **what does `reports/pipeline-activation-wire-compression-report.html`
need that no measurement exists for?** Verified by a 9-agent audit (5 independent inventories,
3 adversarial refutation attempts, 1 completeness critic) over all four report HTMLs, the
archived data tree, both probe tarballs, both log tarballs, and the full git object graph.

Driver: `run_gapfill.sh [html|stage0|stage1|stage2|stage3|all]`, default `html`.

## The answer: 3 runs, not 7

The report draws every figure at runtime from JS config objects — there is no inline `<svg>` —
so a missing cell is a missing array element, and they are enumerable exactly.

| # | stage | label | what it fills | ~time |
|---|---|---|---|---|
| 1 | 0 | `ctl_pp8_bf16` | nothing — baseline validity check, see below | 20 m |
| 2 | 1 | `pp8_fp8` | `depthSafeConfig` fp8 `x:7`, `gsm35EightBitConfig` fp8 `x:7`, `ttftConfig` fp8@7, both latency tables @7 | 25 m |
| 3 | 2 | `lp2_pp8_fp8` | `timeSafeConfig` fp8 series | 25 m |
| 4 | 2 | `lp2_pp8_mxfp8` | `timeSafeConfig` mxfp8 series | 25 m |

The report **already annotates the largest of these gaps itself**: `L1132` carries the chart
label `"FP8 not run at 7 boundaries"`, `L649` says so in prose, `L680` in the figcaption. Run 2
is what lets those three disclaimers be deleted.

## Dropped: the round-2 replicates

`rep2_pp4_{int8,fp8,mxfp8}` are **not** in the default target. This report has **no replication,
variance or reproducibility section at all** — the audit scanned for `replicat`, `reproduc`,
`variance`, `re-run`, `rerun`, `independent`, `second run`, `rep2`, `seed`, `twice`, `run-to-run`,
`error bar`, `standard deviation` and found nothing. Its only uncertainty figure is an analytic
Wilson binomial interval on the n=100 GSM8K sample (`wilsonInterval(percent, n = 100, z = 1.96)`,
`L934`), which is within-run sampling error, not run-to-run variance.

The replication *table* lives in the older `reports/pp_wire_quant_report.html` (line 6429,
pp4 only, bf16/int4/mxfp4/nvfp4). If that file is the target, run `all` instead of `html` and
the three replicates come back — worth doing on the science, since round 2 replicated every
format whose result was dramatic and none whose result was "no effect".

## Stage 0 is not a cell, and still earns its 20 minutes

Every filled cell is plotted as **% change vs the BF16 control**, and that control
(`1.8932072605675765`) was measured on a box that no longer exists. If this box disagrees,
stage 1's fp8 point is drawn against the wrong baseline — the chart goes quietly wrong rather
than visibly empty, which is worse than the gap it fixes.

That number is bit-reproducible on this stack: `main_grid/pp4_bf16` and `ppq8/results/pp8_bf16`
report it identically to the last digit, from different servers on different boxes. And
`git diff a62276c35 HEAD -- python/` is empty, so branch HEAD is byte-identical runtime to what
produced the archived rows. Stage 0 confirms both in 20 minutes, writes `control_verdict.json`,
and continues either way — drift does not invalidate the runs, it just means they must be read
against the new control and the report must say so.

`SKIP_CONTROL=1` skips it.

## Fix these with zero GPU time — already measured, never plotted

The audit found ~50 cells in this category. The ones that touch the same FP8/MXFP8 story:

- **FP8 has no row in either latency table or the TTFT chart at *any* depth**, though
  `main_grid/pp{2,4}_fp8/bench.jsonl` both exist. Droppable straight in:
  ITL `+3.9%` / `+2.5%`; raw `402 ms · 13.92` / `235 ms · 17.92`; TTFT `-1.0%` / `-20.9%`.
  The 1-boundary TTFT is not decorative — it is the only measured case of an 8-bit wire
  *failing* to win on TTFT (vs INT8's `-23.1%`), and omitting it makes the 8-bit latency story
  look cleaner than the data.
- **MXFP8 does have 7-boundary decode-position data** — the 256-step probe, in
  `ppq8_probe_raw.tar.gz`, reported in no HTML. Segments `0.9805 / 0.9831 / 0.9792 / 0.9883`,
  the flattest non-BF16 series measured. If a 256-step series is acceptable in that figure,
  this replaces run 4 for free.
- **Output throughput at 7 boundaries** exists for all six formats and appears in no HTML
  (this report has zero occurrences of "throughput"). It matters: NVFP4, the recommended 4-bit
  format, costs `-16.7%` output throughput at 7 boundaries — *worse* than per-token INT4's
  `-8.3%`, and the penalty grows with depth.
- **All 122B latency** (TTFT/ITL/throughput, both depths, 5 formats) is measured and unreported.

## Operating the box

```bash
# 1. from your laptop — provisions secrets, clone and install in one shot
bash scripts/vast/ppq122/provision_box.sh <ssh-port> <host>

# 2. start the queue detached (it must outlive the ssh session)
ssh -i ~/.ssh/vast_ai -p <port> root@<host> \
  'setsid nohup bash /workspace/sglang-src/scripts/vast/ppq122/run_gapfill.sh html \
   > /workspace/ppq9/logs/queue.log 2>&1 < /dev/null & echo started'

# 3. before destroying the box
bash scripts/vast/ppq122/fetch_gapfill.sh <port> <host>
```

`provision_box.sh` reads `HF_TOKEN` from `~/.config/verl-research/secrets.env` and ships **only
that one variable** to the box over stdin — the earlier flow copied the whole `.env`, handing a
rented third-party machine the Vast API keys, R2 credentials and WANDB key that the run never
touches.

Budget **~3 h**: ~45 min provisioning, ~1.6 h of runs, slack for one retry. Risk is low —
model, codecs, harness and topology are all validated; only the box is new, which is exactly
what stage 0 measures.
