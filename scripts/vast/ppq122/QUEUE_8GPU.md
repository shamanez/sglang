# 8× RTX 5090 run queue

Ordered highest-value-first, so partial box time still buys the best science.
Driver: `run_8gpu.sh [stage1|stage2|stage3|all]`. Every config launches its own
server and the harness asserts the live pp/tp against the label before writing.

**Before stage 1:** copy `reports/data/reference_trajectories_256.json` from the repo
to `$PPQ8/results/reference_trajectories.json`. The pp8 rows must score against the
same frozen references as the published pp2/pp4 rows, or they are not comparable.

---

## Stage 1 — 35B at pp8/tp1: extend the depth axis to 7 boundaries

**The question.** Depth is the only axis on which wire error was shown to accumulate.
We have 1 boundary (pp2) and 3 (pp4); per-token int4 went +11% → +59% NLL and 97% →
64% → 3% GSM8K, while mxfp4/nvfp4 stayed flat. Seven boundaries tests both halves:
does int4's superlinear collapse keep accelerating, and does the microscaled rescue
*hold* at depth, which is what makes it deployable rather than a coincidence.

| # | config | wire | expect |
|---|---|---|---|
| 1 | `pp8_bf16` | none | control; sets the topology's own baseline |
| 2 | `pp8_int8` | int8 | flat (was +0.01% at 3 boundaries) |
| 3 | `pp8_mxfp8` | mxfp8 | flat; confirms no 8-bit format cares about depth |
| 4 | `pp8_int4` | int4 | prediction: catastrophic, GSM8K ~0%, NLL +100%+ |
| 5 | `pp8_mxfp4` | mxfp4 | the real test — does 96–99% survive 7 hops? |
| 6 | `pp8_nvfp4` | nvfp4 | was +0.05% at 3 hops; flat here is the headline claim |

**Stage 1b — 1024-step probe at pp8** (`lp_pp8_*`, 5 configs). The pp2 attempt OOMed
in the scoring path because its last stage held 20 layers; pp8 holds 5, so this
should clear. Recovers the long-horizon × topology cell we lost.

Cost: 6 + 5 configs, ~15 min each ≈ **2.5–3 h**. Risk: **low** — model, codecs, and
harness all validated on 4 GPUs; only the topology is new.

---

## Stage 2 — 122B-FP8 at pp8/tp1: the replication that failed on 4 GPUs

**The question.** Everything so far is one model at one hidden size (2,048). The
granularity mechanism — per-token scales being crushed by outlier channels — predicts
the effect holds or strengthens as hidden size grows. This is the test.

Fit: 118.43 GiB of weights over 8 GPUs ≈ **15 GiB/GPU**, comfortable. The 4-GPU
attempt missed by 18 MB; this is not close. hidden_size 3,072 → **12,288 B/token/
boundary**, 1.5× the 35B payload, so the bandwidth win is proportionally larger.

| # | config | wire |
|---|---|---|
| 1 | `q122_pp8_bf16` | none (also generates this model's own references) |
| 2 | `q122_pp8_int8` | int8 |
| 3 | `q122_pp8_int4` | int4 |
| 4 | `q122_pp8_mxfp4` | mxfp4 |
| 5 | `q122_pp8_nvfp4` | nvfp4 |

Cost: 5 configs, slower load and eval, ~35 min each ≈ **3 h**. Risk: **medium** —
first time this checkpoint has served at all. FP8 weights are orthogonal to the wire
codec (inter-stage tensors stay bf16), so the experiment remains valid.

---

## Stage 3 — 122B-FP8 at pp4/tp2: within-model depth comparison

Same model at 3 boundaries instead of 7, using all 8 GPUs. Without this, any 122B
depth claim is a cross-model comparison; with it, the depth curve is measured inside
one model. Scores against stage 2's references.

Configs: `q122_pp4_{bf16,int8,int4,mxfp4,nvfp4}`. Cost ≈ **3 h**. Risk: low once
stage 2 has served.

---

## Stage 4 — Inkling models: DROPPED (recon verdict, 2026-08-23)

Both `thinkingmachines/Inkling-Small` and `-NVFP4` are unrunnable for this experiment:

- **No PP support in sglang.** `models/inkling.py` builds all 42 layers on every rank
  (`make_layers` without pp params, `inkling.py:611`), its forward has no
  `pp_proxy_tensors`, and `model_runner.py:445-453` hard-asserts `--pp-size > 1`
  against exactly this. Zero pipeline boundaries possible ⇒ no wire to quantize.
- **Inkling-Small bf16 does not fit regardless**: 495.4 GiB weights vs ~238 GiB
  usable across 8×32 GB.
- Inkling-Small-NVFP4 (159 GiB) would fit at tp8/pp1 — but that is 0 boundaries.
- For the record: the NVFP4 weight path itself is *not* blocked on sm_120
  (`is_blackwell_supported` admits capability 12.0; falls to the flashinfer_cutlass
  backend), so weight format was never the blocker — PP support is.

Making Inkling PP-aware means partitioning its Mamba-style sconv cache, reworking the
multimodal embed routine, and handling 8 MTP layers — days of model work, out of scope.

---

## Explicitly dropped

| item | why |
|---|---|
| 122B **bf16** (244 GB) | exceeds 256 GB total once KV cache and CUDA contexts are counted; FP8 is the only viable variant |
| CPU-offload paths | upstream bug for this model family — device mismatch in the Gemma-layernorm weight loader (`layers/layernorm.py:1073`). Not needed at 8 GPUs; worth an upstream issue, not a workaround |
| pp2 1024-step probe | OOMs in the scoring path at 20 layers/stage; stage 1b covers the long-horizon question at pp8 instead |

## Not a run, but the highest-value engineering follow-up

The 4-bit codecs are eager-PyTorch emulations and pay +12% (nvfp4) to +38% (mxfp4)
ITL. That is implementation overhead, not intrinsic wire cost. A fused nvfp4 wire
kernel — the tree already ships NVFP4 quant kernels for GEMM inputs — plus gating
quantization to extend/prefill mode would keep the 21–23% TTFT win and drop the
decode tax. Do this before any deployability claim about 4-bit.
