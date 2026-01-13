# python/sglang/test/test_kvstore_file_p2p_snapshot.py
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from sglang.srt.kv_store.file_kv_store import FileKVStore
from sglang.srt.kv_store.kv_snapshot import (
    export_mha_kv_snapshot,
    import_mha_kv_snapshot,
    pretty_snapshot_brief,
)

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

def _make_mha_pool(device: torch.device) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        64,                 # size
        1,                  # page_size
        torch.float16,      # dtype
        4,                  # head_num
        8,                  # head_dim
        3,                  # layer_num
        str(device),        # device
        False,              # enable_memory_saver
    )

def _get_v_head_dim(pool) -> int:
    return int(getattr(pool, "v_head_dim", getattr(pool, "head_dim")))

@pytest.mark.parametrize("use_cuda", [False, True])
def test_file_kvstore_p2p_mha_snapshot_roundtrip(tmp_path: Path, use_cuda: bool):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0" if use_cuda else "cpu")

    # ---- A side ----
    pool_a = _make_mha_pool(device)
    T = 11
    token_indices_a = torch.tensor([1, 2, 3, 7, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int64)
    assert token_indices_a.numel() == T

    # 模拟 prompt token ids（用于上下文恢复 / 校验）
    token_ids = torch.randint(low=0, high=32000, size=(T,), dtype=torch.int32)

    # 往 pool_a 写入可验证的内容
    # 这里直接写随机 K/V：真实场景是 prefill forward 后 pool 里已经有值
    for l in range(pool_a.layer_num):
        ka = torch.randn((T, pool_a.head_num, pool_a.head_dim), device=device, dtype=pool_a.k_buffer[l].dtype)
        # va = torch.randn((T, pool_a.head_num, pool_a.v_head_dim), device=device, dtype=pool_a.v_buffer[l].dtype)
        vd = _get_v_head_dim(pool_a)
        va = torch.randn((T, pool_a.head_num, vd), device=device, dtype=pool_a.v_buffer[l].dtype)
        pool_a.k_buffer[l].index_copy_(0, token_indices_a.to(device), ka)
        pool_a.v_buffer[l].index_copy_(0, token_indices_a.to(device), va)

    snap = export_mha_kv_snapshot(
        kv_pool=pool_a,
        token_indices=token_indices_a,
        token_ids=token_ids,
        meta_extra={"note": "unit_test_export"},
    )

    # ---- write to store (A -> store) ----
    store = FileKVStore(tmp_path / "kvstore")
    key = f"req_{uuid.uuid4().hex}"
    path = store.put(key, snap)
    print(f"[A] wrote snapshot: key={key} path={path}")
    print(f"[A] {pretty_snapshot_brief(snap)}")

    # ---- B side ----
    pool_b = _make_mha_pool(device)

    # B 侧 token_indices 可以不同（真实 p2p：B 会重新 alloc 一段槽位）
    token_indices_b = torch.tensor([5, 6, 8, 12, 13, 14, 15, 24, 25, 26, 27], dtype=torch.int64)

    loaded = store.get(key, map_location="cpu")
    print(f"[B] loaded snapshot: {pretty_snapshot_brief(loaded)}")

    import_mha_kv_snapshot(
        kv_pool=pool_b,
        token_indices=token_indices_b,
        snapshot=loaded,
        strict=True,
    )

    # 验证：从 A gather 出来的值 == B scatter 写进去后的值
    idx_a = token_indices_a.to(device=device)
    idx_b = token_indices_b.to(device=device)

    for l in range(pool_a.layer_num):
        a_k = pool_a.k_buffer[l].index_select(0, idx_a).cpu()
        b_k = pool_b.k_buffer[l].index_select(0, idx_b).cpu()
        a_v = pool_a.v_buffer[l].index_select(0, idx_a).cpu()
        b_v = pool_b.v_buffer[l].index_select(0, idx_b).cpu()
        torch.testing.assert_close(a_k, b_k, rtol=0, atol=0)
        torch.testing.assert_close(a_v, b_v, rtol=0, atol=0)

    # 顺便检查 token_ids 也一致（上下文恢复/重放需要它）
    assert torch.equal(loaded["token_ids"], token_ids.cpu())
