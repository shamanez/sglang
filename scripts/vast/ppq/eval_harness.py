#!/usr/bin/env python3
"""Evaluation harness for the PP activation wire-quant experiment.

Runs against a live sglang server and writes raw per-item JSON under
--out/<label>/. Metrics per config:
  1. wikitext  — teacher-forced NLL/perplexity on WikiText-2 test windows.
  2. gsm8k     — exact-match accuracy, greedy, 100 test questions.
  3. probe     — per-position NLL / top-1 agreement / top-5 logprobs of a fixed
                 reference trajectory (generated once by the reference config),
                 teacher-forced through this config. Positionwise drift through
                 the GDN recurrent state is the accumulation instrument.
The reference trajectories live at --out/reference_trajectories.json and are
generated only when --make-reference is passed (run that on the baseline).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
CACHE_DIR = "/workspace/models"

N_WIKI_WINDOWS = 24
WIKI_WINDOW = 1024
N_GSM8K = 100
N_PROBE_WIKI = 6
N_PROBE_GSM = 6
PROBE_PROMPT_LEN = 256
PROBE_CONT_LEN = 256
TOPK = 5


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)


def generate(base_url, payload, timeout=600):
    r = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def score_ids(base_url, input_ids, start_len, topk=0):
    """Teacher-forced logprobs of input_ids[start_len:] given the prefix."""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start_len,
    }
    if topk:
        payload["top_logprobs_num"] = topk
    out = generate(base_url, payload)
    meta = out["meta_info"]
    lps = [
        (e[0], e[1])
        for e in meta["input_token_logprobs"]
        if e is not None and e[0] is not None
    ]
    tops = None
    if topk:
        tops = [
            [[t[0], t[1]] for t in pos] if pos else None
            for pos in meta.get("input_top_logprobs") or []
        ]
    return lps, tops


def wiki_windows(tok, n_windows, window, offset=0):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors=None)["input_ids"]
    out = []
    for i in range(n_windows):
        s = offset + i * (window + 1)
        w = ids[s : s + window + 1]
        if len(w) < window + 1:
            break
        out.append(w)
    return out


def gsm8k_items(n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(n):
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"question": ds[i]["question"], "gold": gold})
    return items


def extract_answer(text):
    seg = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", seg)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def chat_ids(tok, question):
    msgs = [
        {
            "role": "user",
            "content": question
            + "\nPlease reason step by step, and put your final answer after '####'.",
        }
    ]
    try:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
    except (TypeError, ValueError):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def run_wikitext(base_url, tok, outdir):
    rows = []
    for w, ids in enumerate(wiki_windows(tok, N_WIKI_WINDOWS, WIKI_WINDOW)):
        lps, _ = score_ids(base_url, ids, start_len=1)
        nll = [-lp for lp, _tid in lps]
        rows.append({"window": w, "n_tokens": len(nll), "sum_nll": sum(nll)})
        print(f"  wikitext window {w}: {len(nll)} toks, mean nll {sum(nll)/len(nll):.4f}")
    total_nll = sum(r["sum_nll"] for r in rows)
    total_tok = sum(r["n_tokens"] for r in rows)
    summary = {"mean_nll": total_nll / total_tok, "n_tokens": total_tok, "windows": rows}
    (outdir / "wikitext.json").write_text(json.dumps(summary))
    print(f"  wikitext mean NLL {summary['mean_nll']:.5f} over {total_tok} tokens")


def run_gsm8k(base_url, tok, outdir):
    rows = []
    n_correct = 0
    for i, item in enumerate(gsm8k_items(N_GSM8K)):
        ids = chat_ids(tok, item["question"])
        t0 = time.time()
        out = generate(
            base_url,
            {
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1024, "temperature": 0.0},
            },
        )
        pred = extract_answer(out["text"])
        ok = pred is not None and pred == item["gold"]
        n_correct += ok
        rows.append(
            {
                "i": i,
                "gold": item["gold"],
                "pred": pred,
                "correct": ok,
                "gen_tokens": out["meta_info"]["completion_tokens"],
                "latency_s": round(time.time() - t0, 2),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  gsm8k {i+1}/{N_GSM8K}: acc so far {n_correct/(i+1):.3f}")
    summary = {"accuracy": n_correct / len(rows), "n": len(rows), "items": rows}
    (outdir / "gsm8k.json").write_text(json.dumps(summary))
    print(f"  gsm8k accuracy {summary['accuracy']:.3f}")


def probe_prompts(tok):
    prompts = []
    for ids in wiki_windows(
        tok, N_PROBE_WIKI, PROBE_PROMPT_LEN, offset=N_WIKI_WINDOWS * (WIKI_WINDOW + 1)
    ):
        prompts.append({"kind": "wikitext", "prompt_ids": ids[: PROBE_PROMPT_LEN]})
    for item in gsm8k_items(N_GSM8K + N_PROBE_GSM)[N_GSM8K:]:
        prompts.append({"kind": "gsm8k", "prompt_ids": chat_ids(tok, item["question"])})
    return prompts


def make_reference(base_url, tok, out_root):
    refs = []
    for p in probe_prompts(tok):
        out = generate(
            base_url,
            {
                "input_ids": p["prompt_ids"],
                "sampling_params": {
                    "max_new_tokens": PROBE_CONT_LEN,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
        )
        cont_ids = [e[1] for e in out["meta_info"]["output_token_logprobs"]]
        refs.append({**p, "cont_ids": cont_ids})
        print(f"  reference {p['kind']}: {len(cont_ids)} tokens")
    (out_root / "reference_trajectories.json").write_text(json.dumps(refs))
    print(f"  wrote {len(refs)} reference trajectories")


def run_probe(base_url, out_root, outdir):
    refs = json.loads((out_root / "reference_trajectories.json").read_text())
    rows = []
    for i, ref in enumerate(refs):
        full = ref["prompt_ids"] + ref["cont_ids"]
        start = len(ref["prompt_ids"])
        lps, tops = score_ids(base_url, full, start_len=start, topk=TOPK)
        # Align: keep exactly the continuation positions.
        lps = lps[-len(ref["cont_ids"]):]
        tops = tops[-len(ref["cont_ids"]):] if tops else None
        rows.append(
            {
                "i": i,
                "kind": ref["kind"],
                "ref_ids": ref["cont_ids"],
                "nll": [-lp for lp, _t in lps],
                "scored_ids": [t for _lp, t in lps],
                "top": tops,
            }
        )
        print(f"  probe {i}: {len(lps)} positions scored")
    (outdir / "probe.json").write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--out", default="/workspace/ppq/results")
    ap.add_argument("--make-reference", action="store_true")
    ap.add_argument("--stages", default="wikitext,gsm8k,probe")
    args = ap.parse_args()

    out_root = Path(args.out)
    outdir = out_root / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()

    info = requests.get(f"{args.base_url}/get_server_info", timeout=30).json()
    (outdir / "server_info.json").write_text(
        json.dumps(
            {
                k: info.get(k)
                for k in ("pp_size", "tp_size", "attention_backend", "version")
            }
            | {"SGLANG_PP_ACTIVATION_WIRE_QUANT": os.environ.get("WIRE_QUANT_LABEL", "")}
        )
    )

    stages = args.stages.split(",")
    if args.make_reference:
        print("[reference]")
        make_reference(args.base_url, tok, out_root)
    if "wikitext" in stages:
        print("[wikitext]")
        run_wikitext(args.base_url, tok, outdir)
    if "gsm8k" in stages:
        print("[gsm8k]")
        run_gsm8k(args.base_url, tok, outdir)
    if "probe" in stages:
        print("[probe]")
        run_probe(args.base_url, out_root, outdir)
    print(f"done: {outdir}")


if __name__ == "__main__":
    main()
