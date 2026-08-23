# PP wire-quant experiment — archived results

Raw measurements for the report in `../pp_wire_quant_report.html`. Kept here because
the experiments ran on a rented Vast.ai box whose disk does not outlive the rental.

| path | contents |
|---|---|
| `main_grid/<label>/` | the 14 configs: pp2/pp4 × bf16, int8, fp8, mxfp8, int4, mxfp4, nvfp4 |
| `replicates/rep2_<label>/` | independent re-runs of 4 headline pp4 cells (fresh servers) |
| `report_aggregates.json` | one summary row per config, as plotted |
| `probe_segment_aggregates.json` | positionwise probe reduced to per-segment agreement + NLL (256-step probe in quarters, 1024-step probe in eighths) |

Per config: `wikitext.json` (teacher-forced NLL), `gsm8k.json` (per-item predictions —
re-grade numerically, not by string equality), `bench.jsonl` (bench_serving),
`server_info.json` (the actual pp/tp the results were measured on).

Raw `probe.json` files (per-step logprobs and top-k, ~600 KB per config, 21 MB total)
are not archived; `probe_segment_aggregates.json` holds every number the report uses.

Not archived because the runs failed: the Qwen3.5-122B-A10B-FP8 wire series (weights
did not fit in 4×32 GB) and the pp2 1024-step probes (OOM in the scoring path). See
the report's "what could not be run" section.
