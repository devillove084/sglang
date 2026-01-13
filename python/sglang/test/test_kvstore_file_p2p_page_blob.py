# python/sglang/test/test_kvstore_file_p2p_page_blob.py
from __future__ import annotations

from pathlib import Path

import torch

from sglang.srt.kv_store.file_kv_store import FileKVStore
from sglang.srt.kv_store.kv_snapshot_page_blob import export_kv_pages_to_files, import_kv_pages_from_files
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def _make_pool(device: str) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=256,
        page_size=16,
        dtype=torch.float16,
        head_num=4,
        head_dim=8,
        layer_num=3,
        device=device,
        enable_memory_saver=False,
    )


@torch.no_grad()
def test_page_blob_roundtrip_mha(tmp_path: Path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool_a = _make_pool(dev)
    pool_b = _make_pool(dev)

    # 构造一些“已用 token slots”的内容：前 80 tokens 填随机
    T = 80
    for l in range(pool_a.layer_num):
        pool_a.k_buffer[l][:T].uniform_(-1, 1)
        pool_a.v_buffer[l][:T].uniform_(-1, 1)

    # 假设这次请求占用了 token_indices = [0..79]，那么 page_indices = [0..4] (page_size=16)
    token_indices = torch.arange(T, dtype=torch.int64)
    page_size = pool_a.page_size
    page_indices = torch.unique(token_indices // page_size).cpu()

    store = FileKVStore(tmp_path / "kvstore")
    key = "rid_123"

    export_kv_pages_to_files(
        kv_pool=pool_a,
        page_indices=page_indices,
        store=store,
        key=key,
        token_ids=torch.arange(32, dtype=torch.int32),
    )

    # B侧：为了测试简单，dst pages 就用同一组 pages
    import_kv_pages_from_files(
        kv_pool=pool_b,
        dst_page_indices=page_indices,
        store=store,
        key=key,
        strict=True,
    )

    # 校验：这些 pages 覆盖的 token 范围内容一致
    max_page = int(page_indices.max().item())
    t0 = 0
    t1 = (max_page + 1) * page_size

    for l in range(pool_a.layer_num):
        assert torch.allclose(pool_a.k_buffer[l][t0:t1], pool_b.k_buffer[l][t0:t1])
        assert torch.allclose(pool_a.v_buffer[l][t0:t1], pool_b.v_buffer[l][t0:t1])
