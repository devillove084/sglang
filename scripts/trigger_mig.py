import json, re, threading, time
import requests

A = "http://127.0.0.1:52000"
B = "http://127.0.0.1:52001/file_p2p"   # 你的 dst_http 就是这个

gen_url = f"{A}/generate"
abandon_url = f"{A}/file_p2p/abandon_req"

payload = {
  "text": "Write an extremely long story about migrating requests. Keep going and do not conclude.",
  "max_new_tokens": 8192,
  "stream": True
}

rid_holder = {"rid": None}

def do_abandon(rid: str):
  time.sleep(0.05)  # 给 scheduler 一点点时间把 rid 注册进 rid_to_state（可调大）
  r = requests.post(abandon_url, json={
    "rid": rid,
    "dst_http": B,
    "reason": "p2p_migration_test",
    "abort_local": False
  }, timeout=10)
  print("\n[ABANDON RESP]", r.status_code, r.text)

with requests.post(gen_url, json=payload, stream=True, timeout=30) as resp:
  resp.raise_for_status()

  for raw in resp.iter_lines(decode_unicode=True):
    if not raw:
      continue
    line = raw.strip()
    if line.startswith("data:"):
      line = line[len("data:"):].strip()

    # 1) 尝试 JSON 解析
    rid = None
    try:
      obj = json.loads(line)
      mi = obj.get("meta_info") or {}
      rid = mi.get("id") or obj.get("id")
    except Exception:
      # 2) fallback：正则抓 32hex 的 id
      m = re.search(r'"meta_info"\s*:\s*\{.*?"id"\s*:\s*"([0-9a-fA-F]{32})"', line)
      if not m:
        m = re.search(r'"id"\s*:\s*"([0-9a-fA-F]{32})"', line)
      if m:
        rid = m.group(1)

    if rid and rid_holder["rid"] is None:
      rid_holder["rid"] = rid
      print("[RID FOUND]", rid)
      threading.Thread(target=do_abandon, args=(rid,), daemon=True).start()

    # 可选：打印少量流（避免刷屏）
    print(line[:200])

  print("\n[GEN STREAM END]")
