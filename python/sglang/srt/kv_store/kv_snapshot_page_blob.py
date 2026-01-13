# python/sglang/srt/mem_cache/kv_snapshot_page_blob.py
from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# -----------------------------
# helpers
# -----------------------------

def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    # 覆盖常见 dtype；你后面如果用 fp8/fp4，建议走 store_dtype=uint8 的那条逻辑
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.bfloat16:
        # numpy 没有原生 bfloat16（有些版本有），这里退化为 uint16 存 raw bits
        return np.uint16
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.int32:
        return np.int32
    if dtype == torch.int64:
        return np.int64
    if dtype == torch.uint8:
        return np.uint8
    raise TypeError(f"Unsupported dtype for raw blob: {dtype}")


def _group_consecutive(sorted_pages: List[int]) -> List[Tuple[int, int]]:
    """
    输入升序 pages，输出 runs: [(start, length_pages), ...]
    """
    if not sorted_pages:
        return []
    runs: List[Tuple[int, int]] = []
    s = sorted_pages[0]
    prev = s
    length = 1
    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
            length += 1
        else:
            runs.append((s, length))
            s = p
            prev = p
            length = 1
    runs.append((s, length))
    return runs


def _ensure_int_list(page_indices: Any) -> List[int]:
    if isinstance(page_indices, torch.Tensor):
        page_indices = page_indices.detach().cpu().to(torch.int64).tolist()
    elif isinstance(page_indices, np.ndarray):
        page_indices = page_indices.astype(np.int64).tolist()
    else:
        page_indices = list(page_indices)
    return [int(x) for x in page_indices]


def _infer_arch_and_layer_bufs(kv_pool: Any) -> Tuple[str, List[torch.Tensor]]:
    """
    返回 (arch, layer_bufs)
    - arch="mha": layer_bufs = [k_layer0..k_layerL-1, v_layer0..v_layerL-1]
    - arch="mla": layer_bufs = [kv_layer0..kv_layerL-1]  (融合KV)
    """
    if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
        kbuf = getattr(kv_pool, "k_buffer")
        vbuf = getattr(kv_pool, "v_buffer")
        if isinstance(kbuf, list) and isinstance(vbuf, list):
            return "mha", list(kbuf) + list(vbuf)

    # MLA 常见命名：kv_buffer 或 kv_data（你后续如果发现名字不同，往这里加即可）
    for name in ("kv_buffer", "kv_data", "kv_buf"):
        if hasattr(kv_pool, name):
            buf = getattr(kv_pool, name)
            if isinstance(buf, list):
                return "mla", list(buf)

    raise AttributeError(
        "Cannot infer KV pool buffers. Expected (k_buffer,v_buffer) for MHA "
        "or (kv_buffer/kv_data) for MLA."
    )


def _layer_num_from_pool(kv_pool: Any, arch: str, layer_bufs: List[torch.Tensor]) -> int:
    if hasattr(kv_pool, "layer_num"):
        return int(getattr(kv_pool, "layer_num"))
    # fallback：MHA 是 2*L
    if arch == "mha":
        return len(layer_bufs) // 2
    return len(layer_bufs)


def _pool_page_size(kv_pool: Any) -> int:
    if not hasattr(kv_pool, "page_size"):
        raise AttributeError("kv_pool.page_size is required for page-based snapshot")
    return int(getattr(kv_pool, "page_size"))


def _pool_store_dtype(kv_pool: Any) -> torch.dtype:
    # SGLang 有些实现区分 dtype / store_dtype
    if hasattr(kv_pool, "store_dtype"):
        return getattr(kv_pool, "store_dtype")
    if hasattr(kv_pool, "dtype"):
        return getattr(kv_pool, "dtype")
    # fallback 从 buffer dtype 推断
    arch, bufs = _infer_arch_and_layer_bufs(kv_pool)
    return bufs[0].dtype


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


# -----------------------------
# Blob format
# -----------------------------

@dataclass
class PageBlobLayout:
    """
    描述 .bin 的布局：
    bin = concat over layers:
            concat over runs:
               raw bytes of buf[token_start:token_end]  (token_end-token_start = run_pages*page_size)
    """
    arch: str                    # "mha" or "mla"
    page_size: int
    pages: List[int]             # sorted unique pages
    runs: List[Tuple[int, int]]  # (start_page, len_pages)
    layer_offsets: List[int]     # per "layer_buf" offset into bin
    chunk_bytes_per_run: List[int]  # per run bytes length (same for all layers with same per-token shape)
    per_token_shape: Tuple[int, ...]
    store_dtype: str             # string repr
    bytes_per_token: int
    bytes_per_page: int
    layer_buf_count: int         # mha: 2*L, mla: L


# -----------------------------
# Export
# -----------------------------

@torch.no_grad()
def export_kv_pages_to_files(
    *,
    kv_pool: Any,
    page_indices: Any,
    store: Any,        # FileKVStore
    key: str,
    token_ids: Optional[torch.Tensor] = None,   # 可选：prompt token ids（只做校验/调试）
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    把 kv_pool 的指定 pages 导出到：
      - {key}.meta.pt  (torch.save)
      - {key}.bin      (raw bytes)
    """
    arch, layer_bufs = _infer_arch_and_layer_bufs(kv_pool)
    page_size = _pool_page_size(kv_pool)
    store_dtype = _pool_store_dtype(kv_pool)
    layer_num = _layer_num_from_pool(kv_pool, arch, layer_bufs)

    pages = sorted(set(_ensure_int_list(page_indices)))
    if len(pages) == 0:
        raise ValueError("page_indices is empty")

    runs = _group_consecutive(pages)

    # 每个 token 的 shape（除第一维 tokens）
    per_token_shape = tuple(layer_bufs[0].shape[1:])
    bytes_per_token = int(torch.tensor([], dtype=store_dtype).element_size())
    # 兼容 bfloat16 被存成 uint16 raw bits 的情况：
    if store_dtype == torch.bfloat16:
        bytes_per_token = 2
    bytes_per_page = bytes_per_token * int(np.prod(per_token_shape)) * page_size

    # 计算每个 run 写入多少 bytes
    chunk_bytes_per_run: List[int] = []
    for _, run_pages in runs:
        chunk_bytes_per_run.append(bytes_per_page * run_pages)

    # 写 bin：逐层写（layer_buf 顺序），每层按 runs 顺序写
    bin_path = store.get_bytes_path(key, suffix=".bin")
    tmp = Path(str(bin_path) + ".tmp")
    layer_offsets: List[int] = []
    offset = 0

    with open(tmp, "wb") as f:
        for lb in layer_bufs:
            layer_offsets.append(offset)
            # 强一致校验：shape/dtype
            if tuple(lb.shape[1:]) != per_token_shape:
                raise ValueError(f"per_token_shape mismatch: {tuple(lb.shape[1:])} vs {per_token_shape}")
            if lb.dtype != store_dtype:
                # 如果你 pool 里 store_dtype=uint8（fp8/量化），这里应该一致；
                # 如果不一致，说明你拿到的是 compute dtype，需要改用 store buffer。
                raise ValueError(f"dtype mismatch: lb.dtype={lb.dtype} store_dtype={store_dtype}")

            # runs copy out
            for start_page, run_pages in runs:
                t0 = start_page * page_size
                t1 = (start_page + run_pages) * page_size
                # 连续 slice，避免 token 索引 gather
                slab = lb[t0:t1].contiguous()
                slab_cpu = slab.to("cpu", non_blocking=False)

                # bfloat16 作为 uint16 raw bits 存
                if slab_cpu.dtype == torch.bfloat16:
                    slab_cpu = slab_cpu.view(torch.uint16)

                # 写 raw bytes
                f.write(slab_cpu.numpy().tobytes(order="C"))
                offset += _tensor_nbytes(slab_cpu)

    # 原子 rename
    import os
    os.replace(tmp, bin_path)

    layout = PageBlobLayout(
        arch=arch,
        page_size=page_size,
        pages=pages,
        runs=runs,
        layer_offsets=layer_offsets,
        chunk_bytes_per_run=chunk_bytes_per_run,
        per_token_shape=per_token_shape,
        store_dtype=str(store_dtype),
        bytes_per_token=bytes_per_token,
        bytes_per_page=bytes_per_page,
        layer_buf_count=len(layer_bufs),
    )

    meta: Dict[str, Any] = {
        "version": 1,
        "arch": arch,                       # "mha" / "mla"
        "layer_num": layer_num,
        "page_size": page_size,
        "store_dtype": layout.store_dtype,
        "per_token_shape": per_token_shape,
        "pages": pages,
        "runs": runs,
        "layer_offsets": layer_offsets,
        "chunk_bytes_per_run": chunk_bytes_per_run,
        "bytes_per_page": bytes_per_page,
        "layer_buf_count": layout.layer_buf_count,
        "bin_suffix": ".bin",
    }

    if token_ids is not None:
        token_ids_cpu = token_ids.detach().to("cpu").contiguous()
        # 不建议把全量 token_ids 都塞进去（大 prompt 会很大），这里先提供两种：
        # 1) 保存全量（仅测试/调试）
        meta["token_ids"] = token_ids_cpu
        # 2) 保存 hash（更推荐）
        meta["token_ids_hash"] = int(torch.sum(token_ids_cpu.to(torch.int64)).item())

    if meta_extra:
        meta.update(meta_extra)

    # 写 meta
    store.put(key + ".meta", meta)
    return meta


# -----------------------------
# Import
# -----------------------------

@torch.no_grad()
def import_kv_pages_from_files(
    *,
    kv_pool: Any,
    dst_page_indices: Any,   # B侧已经分配好的 pages，长度必须等于 meta["pages"]
    store: Any,              # FileKVStore
    key: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    从 {key}.meta.pt + {key}.bin 读取，并写回 kv_pool 的对应 dst_page_indices pages。
    """
    meta = store.get(key + ".meta", map_location="cpu")
    bin_path = store.get_bytes_path(key, suffix=meta.get("bin_suffix", ".bin"))

    arch, layer_bufs = _infer_arch_and_layer_bufs(kv_pool)
    page_size = _pool_page_size(kv_pool)
    store_dtype = _pool_store_dtype(kv_pool)

    if strict:
        if int(meta.get("version", -1)) != 1:
            raise ValueError(f"Unsupported version: {meta.get('version')}")
        if meta.get("arch") != arch:
            raise ValueError(f"arch mismatch: meta={meta.get('arch')} pool={arch}")
        if int(meta.get("page_size")) != page_size:
            raise ValueError(f"page_size mismatch: meta={meta.get('page_size')} pool={page_size}")
        if str(store_dtype) != meta.get("store_dtype"):
            raise ValueError(f"store_dtype mismatch: meta={meta.get('store_dtype')} pool={store_dtype}")

    src_pages: List[int] = [int(x) for x in meta["pages"]]
    dst_pages: List[int] = _ensure_int_list(dst_page_indices)
    if len(dst_pages) != len(src_pages):
        raise ValueError(f"dst_page_indices length mismatch: dst={len(dst_pages)} src={len(src_pages)}")

    runs: List[Tuple[int, int]] = [tuple(x) for x in meta["runs"]]
    layer_offsets: List[int] = [int(x) for x in meta["layer_offsets"]]
    chunk_bytes_per_run: List[int] = [int(x) for x in meta["chunk_bytes_per_run"]]
    per_token_shape = tuple(meta["per_token_shape"])
    bytes_per_page = int(meta["bytes_per_page"])

    # mmap bin
    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        try:
            # 按 layer_buf 顺序读写
            for li, lb in enumerate(layer_bufs):
                if tuple(lb.shape[1:]) != per_token_shape:
                    raise ValueError(f"per_token_shape mismatch on import: {tuple(lb.shape[1:])} vs {per_token_shape}")
                if lb.dtype != store_dtype:
                    raise ValueError(f"dtype mismatch on import: lb.dtype={lb.dtype} store_dtype={store_dtype}")

                layer_base = layer_offsets[li]
                cursor = layer_base

                # dst_pages 按“src_runs的页数序列”切片消耗：每个 run 消耗 run_pages 个 dst pages
                dp_cursor = 0

                for run_i, (_, run_pages) in enumerate(runs):
                    chunk_nbytes = chunk_bytes_per_run[run_i]
                    # 该 run 覆盖的 dst pages 列表
                    dst_slice_pages = dst_pages[dp_cursor : dp_cursor + run_pages]
                    dp_cursor += run_pages

                    np_dtype = _torch_dtype_to_numpy(store_dtype)
                    if store_dtype == torch.bfloat16:
                        np_dtype = np.uint16

                    itemsize = np.dtype(np_dtype).itemsize
                    assert chunk_nbytes % itemsize == 0
                    count = chunk_nbytes // itemsize

                    arr_view = np.frombuffer(mm, dtype=np_dtype, count=count, offset=cursor)  # view of mmap
                    cursor += chunk_nbytes

                    arr = arr_view.copy()   # detach from mmap
                    del arr_view            # <<< 关键：释放对 mmap 的引用

                    slab_cpu = torch.from_numpy(arr).reshape(run_pages * page_size, *per_token_shape)
                    if store_dtype == torch.bfloat16:
                        slab_cpu = slab_cpu.view(torch.bfloat16)

                    dst_is_consecutive = all(
                        dst_slice_pages[i] + 1 == dst_slice_pages[i + 1]
                        for i in range(len(dst_slice_pages) - 1)
                    )
                    if dst_is_consecutive:
                        start_page = dst_slice_pages[0]
                        t0 = start_page * page_size
                        t1 = (start_page + run_pages) * page_size
                        lb[t0:t1].copy_(slab_cpu.to(device=lb.device, non_blocking=False))
                    else:
                        # per page scatter
                        for j, dp in enumerate(dst_slice_pages):
                            t0 = dp * page_size
                            t1 = (dp + 1) * page_size
                            slab_piece = slab_cpu[j * page_size : (j + 1) * page_size]
                            lb[t0:t1].copy_(slab_piece.to(device=lb.device, non_blocking=False))

        finally:
            mm.close()

    return meta
