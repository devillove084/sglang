# python/sglang/test/test_pd_kv_recovery.py
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _popen(cmd, env, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "wb")
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p, f


def _kill_proc(p: subprocess.Popen, timeout_s=10):
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        p.terminate()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        p.kill()


def _tail_text(path: Path, max_bytes: int = 20_000) -> str:
    if not path.exists():
        return "<log file not found>"
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return repr(data)


def _wait_http_ready(base_url: str, proc: subprocess.Popen, log_file: Path, timeout_s: int = 600):
    """
    - 进程挂了：立刻抛错并带上 log tail
    - 否则：轮询 /health、/get_model_info、/ 直到能连上
    """
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"Server exited early (rc={rc}) for {base_url}\n"
                f"Command log tail:\n{_tail_text(log_file)}"
            )
        for path in ("/health", "/get_model_info", "/"):
            try:
                r = requests.get(base_url + path, timeout=2)
                if r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
        time.sleep(0.2)
    raise RuntimeError(
        f"Server not ready: {base_url}, last_err={last_err}\n"
        f"Command log tail:\n{_tail_text(log_file)}"
    )


def _call_generate(base_url: str, text: str, temperature: float = 0.0, max_new_tokens: int = 64):
    url = base_url + "/generate"
    payload = {
        "text": text,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens},
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "generated_text" in data and isinstance(data["generated_text"], str):
            return data["generated_text"]
    return json.dumps(data, ensure_ascii=False)


@pytest.mark.skipif(os.environ.get("SGLANG_TEST_MODEL") is None, reason="set SGLANG_TEST_MODEL to run this test")
def test_pd_disagg_kv_recovery_matches_monolithic(tmp_path: Path):
    """
    1) 先启动 Prefill(A)+Decode(B)+Router，走 PD disagg，拿到输出 out_pd
    2) 关闭 PD 三个进程
    3) 再启动单机 server，拿到输出 out_mono
    4) 比较 out_pd == out_mono
    """
    import torch
    if (not torch.cuda.is_available()) or torch.cuda.device_count() < 1:
        pytest.skip("PD disaggregation smoke test needs >= 1 CUDA GPUs on the same machine.")

    model = os.environ["SGLANG_TEST_MODEL"]

    pd_backend = os.environ.get("SGLANG_PD_TRANSFER_BACKEND", "nixl").strip()

    prefill_port = _find_free_port()
    decode_port = _find_free_port()
    router_port = _find_free_port()
    mono_port = _find_free_port()

    prefill_url = f"http://127.0.0.1:{prefill_port}"
    decode_url = f"http://127.0.0.1:{decode_port}"
    router_url = f"http://127.0.0.1:{router_port}"
    mono_url = f"http://127.0.0.1:{mono_port}"

    logs = f"/tmp/pd_logs"
    env = os.environ.copy()

    common_det_flags = [
        "--disable-radix-cache",
        "--enable-deterministic-inference",
        "--disable-cuda-graph",
        "--context-length", "1024",
        "--max-running-requests", "1",
    ]
    # ---- Start Prefill worker (A) ----
    prefill_log = logs / "prefill_A.log"
    prefill_cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(prefill_port),
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", pd_backend,
        "--base-gpu-id", "0",
        "--mem-fraction-static", "0.4",
        *common_det_flags,
    ]
    prefill_p, prefill_f = _popen(prefill_cmd, env, prefill_log)

    # ---- Start Decode worker (B) ----
    decode_log = logs / "decode_B.log"
    decode_cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(decode_port),
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", pd_backend,
        "--base-gpu-id", "0",
        "--mem-fraction-static", "0.6",
        *common_det_flags,
    ]
    decode_p, decode_f = _popen(decode_cmd, env, decode_log)

    # ---- Start Router ----
    router_log = logs / "router.log"
    router_cmd = [
        "python", "-m", "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill", prefill_url,
        "--decode", decode_url,
        "--host", "127.0.0.1",
        "--port", str(router_port),
    ]
    router_p, router_f = _popen(router_cmd, env, router_log)

    out_pd = None
    try:
        _wait_http_ready(prefill_url, prefill_p, prefill_log)
        _wait_http_ready(decode_url, decode_p, decode_log)
        _wait_http_ready(router_url, router_p, router_log)

        prompt = "Explain KV cache recovery in one paragraph. "
        out_pd = _call_generate(router_url, prompt, temperature=0.0, max_new_tokens=64)

    finally:
        for p in (router_p, decode_p, prefill_p):
            _kill_proc(p)
        for f in (router_f, decode_f, prefill_f):
            try:
                f.close()
            except Exception:
                pass

    # ---- Start Monolithic baseline (after PD is down) ----
    mono_log = logs / "monolithic.log"
    mono_cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(mono_port),
        "--base-gpu-id", "0",
        *common_det_flags,
    ]
    mono_p, mono_f = _
