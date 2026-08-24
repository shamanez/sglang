# Gap-fill run queue — 8x RTX 5090

Three holes in the published 35B matrix, all of them the per-token **fp8** and
microscaled **mxfp8** wires at depth. Ordered highest-value-first, so partial box
time still buys the best science. Driver: `run_gapfill.sh [stage0|stage1|stage2|stage3|all]`.

| gap | what is missing | stage |
|---|---|---|
| core quality grid | fp8 at 7 boundaries — the only empty cell in the 1/3/7-boundary matrix | 1 |
| 1024-step decode probe | fp8 and mxfp8 at 7 boundaries | 2 |
| independent replication | int8, fp8, mxfp8 at 3 boundaries (round 2 covered only bf16, int4, mxfp4, nvfp4) | 3 |

---

## Before anything: the two things that make the numbers comparable

**1. Frozen references.** Every 35B row in the report scores against one frozen set
of reference trajectories. `bootstrap_gapfill.sh` copies them out of the repo
(`reports/data/reference_trajectories_256.json` and `..._1024_pp8.json`) into the
results root. Regenerating them on the new box would produce a self-consistent set
of numbers that cannot be put in the same table as the published ones.

**2. The runtime is byte-identical.** The archived pp8 results carry
`version 0.0.0.dev16932+ga62276c35`, and `git diff a62276c35 HEAD -- python/` is
empty — every commit on this branch since then has touched only `reports/` and
`scripts/`. So building branch HEAD reproduces the exact code that produced them.
`bootstrap_gapfill.sh` re-checks this on the box and warns if `python/` has moved,
rather than trusting a pinned hash that any later commit would invalidate.

---

## Stage 0 — box reproducibility control (`ctl_pp8_bf16`, ~20 min)

**Why this runs first.** The new cells are measured on a new physical box, but every
delta they feed is against anchors from a box that no longer exists. bf16
teacher-forced NLL is bit-deterministic on this stack — pp4 and pp8, four separate
servers, all returned `1.8932072605675765` to the last digit — so re-measuring it is
a 20-minute test of whether that determinism survives the box change.

- **bit-identical** → every new cell drops straight into the published tables.
- **drift** → the new cells are read against *this* control, and the report says so.
  Not a failure; the runs stay valid. The driver warns, writes
  `control_verdict.json`, and continues.

GSM8K also re-measures the accuracy resolution (archived control: 96%; decode
batching nondeterminism moves this ±1 point, which is the practical floor on any
accuracy claim).

Skippable with `SKIP_CONTROL=1`. Not recommended — it is the cheapest stage and it
is what licenses every comparison after it.

## Stage 1 — `pp8_fp8`, the priority run (~25 min)

35B, bf16 weights, **fp8 wire, pp8/tp1 = 7 boundaries**. Full stage set —
WikiText NLL, GSM8K, 256-step probe, bench_serving — so the row matches the other
six pp8 rows field for field.

fp8's shallower rows were quality-free (**-0.005%** NLL at 1 boundary, **+0.06%** at
3), and the other two 8-bit wires already hold at 7 (int8 **+0.29%** / 99% GSM8K,
mxfp8 **+0.08%** / 96%). So the expected result is flat, and that is precisely the
value: the claim on the table is *no 8-bit format cares about
pipeline depth*, and fp8 is the one format that has never been asked at depth. A
flat row completes it; a non-flat row is the most interesting result of the campaign.

Watch also the **TTFT**: at 1–3 boundaries fp8's Triton per-token quant kernel ate
most of its latency win (402 ms vs mxfp8's 317 ms at pp2), and at 7 boundaries every
format converged to 181–193 ms. This row says whether fp8 converges with them.

## Stage 2 — long-horizon probe, `lp2_pp8_{fp8,mxfp8}` (~50 min)

1024 decode steps at 7 boundaries, against the same frozen 1024-step references as
the published `lp2_pp8_*` set (bf16, int8, int4, mxfp4, nvfp4). This is the
temporal-accumulation instrument: the finding it supports is that temporal
accumulation *does not exist* at any format or depth, and a negative result is worth
only as much as its coverage.

Server args `--mem-fraction-static 0.75 --disable-cuda-graph` are **not** a codec
workaround. Scoring a ~1300-token sequence materializes a `[T, 248k-vocab]` fp32
logprob transient outside the static pool; at 0.94 the first attempt OOMed inside
`_row_logsumexp_topk_kernel`. It is a harness scoring-path cost, and it is why the
pp2 long probes were lost entirely.

## Stage 3 — replication round 2 completion, `rep2_pp4_{int8,fp8,mxfp8}` (~55 min)

3 boundaries, quality only (no bench). Round 2 replicated bf16, int4, mxfp4, nvfp4 —
that is, every format whose result was *dramatic*, and none of the three whose result
was *"no effect"*. An unreplicated null result is exactly the kind that quietly turns
out to be a harness artifact, so these are the replicates that carry the most weight.

pp4/tp1 leaves 4 of the 8 GPUs idle. Running two configs side by side would halve
this stage, but `run_config.sh` hardcodes port 30000 and `pkill`s every sglang
process, so it would need per-config ports and device masks. Not worth the risk of
mislabeling a row for 25 minutes.

---

## Operating the box

```bash
# 1. on your laptop — the box needs HF_TOKEN for the gated Qwen weights
scp -i ~/.ssh/vast_ai -P <port> ~/Documents/sglang/.env root@<host>:/workspace/.env

# 2. on the box
git clone -b research/pp-activation-int8 https://github.com/shamanez/sglang.git /workspace/sglang-src
bash /workspace/sglang-src/scripts/vast/ppq122/bootstrap_gapfill.sh      # ~45 min

# 3. detached, or it dies with the ssh session (this cost several reruns last time)
setsid nohup bash /workspace/sglang-src/scripts/vast/ppq122/run_gapfill.sh all \
    > /workspace/ppq9/logs/queue.log 2>&1 < /dev/null &

tail -f /workspace/ppq9/logs/queue.log     # ends at GAPFILL_ALL_DONE

# 4. on your laptop, BEFORE destroying the box
bash scripts/vast/ppq122/fetch_gapfill.sh <port> <host>
```

`fetch_gapfill.sh` lands each result at the path `build_8gpu_section.py` already
reads, splits the raw 256-step `probe.json` into a tarball the way the earlier
campaigns are archived, and regenerates `reports/data/section_8gpu.html`.

Budget **~4 h** of box time: ~45 min provisioning, ~2.5 h of runs, slack for one
retry. Risk is low throughout — model, codecs, harness, and topology are all
validated; only the box is new, which is what stage 0 measures.

## Deliberately not in this queue

| item | why |
|---|---|
| fp8 / mxfp8 on the **122B** | the 122B series was run as bf16/int8/int4/mxfp4/nvfp4 by design — it exists to test the *granularity* mechanism at a larger hidden size, and its 8-bit anchor is int8. Adding fp8 there is a different question (and another ~35 min/config on a checkpoint whose fit is the risky part) |
| a fused nvfp4 wire kernel | still the highest-value engineering follow-up (the 4-bit codecs are eager-torch emulations paying +12% to +38% ITL), but it is implementation work, not a run |
| parallel pp4 configs | needs per-config ports and device masks in `run_config.sh`; see stage 3 |
