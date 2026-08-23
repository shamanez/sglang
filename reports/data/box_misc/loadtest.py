import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-35B-A3B"
CONC, MAXTOK = 16, 256


def one(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Explain pipeline parallelism, variation {i}."}],
        "max_tokens": MAXTOK,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return d["usage"]["completion_tokens"], time.time() - t0


t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, range(CONC)))
wall = time.time() - t0

toks = sum(r[0] for r in res)
lats = sorted(r[1] for r in res)
print(f"concurrency      {CONC}")
print(f"requests ok      {len(res)}")
print(f"completion toks  {toks}")
print(f"wall             {wall:.2f}s")
print(f"aggregate        {toks / wall:.1f} tok/s")
print(f"latency p50/max  {lats[len(lats)//2]:.2f}s / {lats[-1]:.2f}s")
