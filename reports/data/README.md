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

## Box archive (Vast.ai instance torn down 2026-08-23)

| path | contents |
|---|---|
| `probe_raw.tar.gz` | raw per-step probe data: 14 configs at 256 steps + 5 at 1024 steps (33 MB uncompressed) |
| `run_logs.tar.gz` | 81 run/eval/bench/serve logs, including the 122B fit-test failures and the CPU-offload device-mismatch traceback |
| `reference_trajectories_256.json` | frozen baseline greedy trajectories every 256-step probe scored against |
| `reference_trajectories_1024.json` | same for the long-horizon probe |
| `box_misc/` | dev-install log and the box's loadtest script |

Launch and driver scripts recovered from the box live in `../../scripts/vast/`
(`serve_qwen36.sh`, `download_model.sh`) and `../../scripts/vast/ppq122/`
(parametrized harness variant, 122B serve script, overnight queue drivers).
