# python/sglang/srt/mem_cache/kv_snapshot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


@dataclass
class KVSnapshot:
    """
    约定的 snapshot 格式（V1）
    - arch: "mha"（先只做最常见的 MHATokenToKVPool）
    - token_ids: prompt token ids（用于“上下文恢复”的一致性校验/重放）
    - k, v: shape = [L, T, H, D] (或 v 的 Dv)
    """
    version: int
    arch: str
    token_ids: torch.Tensor            # [T] int32/int64 (CPU)
    k: torch.Tensor                    # [L, T, H, D] on CPU
    v: torch.Tensor                    # [L, T, H, Dv] on CPU
    meta: Dict[str, Any]

    def nbytes(self) -> int:
        return _tensor_nbytes(self.token_ids) + _tensor_nbytes(self.k) + _tensor_nbytes(self.v)


def export_mha_kv_snapshot(
    *,
    kv_pool: Any,
    token_indices: torch.Tensor,
    token_ids: torch.Tensor,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从 MHATokenToKVPool 导出 snapshot。
    kv_pool 需要具备属性：
      - layer_num
      - k_buffer: List[Tensor] each [N, H, D]
      - v_buffer: List[Tensor] each [N, H, Dv]
      - store_dtype / dtype（有些实现会区分）
      - head_num/head_dim/v_head_dim（可选，用于meta）
    token_indices: [T]，指向 kv_pool buffer 的 token 槽位索引
    token_ids: [T]，prompt token ids（用于 B 恢复上下文）
    """
    if token_indices.dtype not in (torch.int32, torch.int64):
        token_indices = token_indices.to(torch.int64)

    # 强制在同设备上 gather，然后转 CPU
    device = kv_pool.k_buffer[0].device
    idx = token_indices.to(device=device, dtype=torch.int64)

    k_layers = []
    v_layers = []
    layer_num = int(getattr(kv_pool, "layer_num", len(kv_pool.k_buffer)))
    for l in range(layer_num):
        k_l = kv_pool.k_buffer[l].index_select(0, idx).contiguous()
        v_l = kv_pool.v_buffer[l].index_select(0, idx).contiguous()
        k_layers.append(k_l)
        v_layers.append(v_l)

    k = torch.stack(k_layers, dim=0).to("cpu", non_blocking=False)
    v = torch.stack(v_layers, dim=0).to("cpu", non_blocking=False)

    token_ids_cpu = token_ids.to("cpu").contiguous()

    meta = {
        "version": 1,
        "arch": "mha",
        "layer_num": layer_num,
        "token_count": int(token_ids_cpu.numel()),
        "k_shape_per_layer": tuple(k_layers[0].shape),
        "v_shape_per_layer": tuple(v_layers[0].shape),
        "dtype": str(k_layers[0].dtype),
        "device_from": str(device),
    }
    # 尽可能补齐一些常用字段（如果 pool 有）
    for name in ("head_num", "head_dim", "v_head_dim", "page_size"):
        if hasattr(kv_pool, name):
            try:
                meta[name] = int(getattr(kv_pool, name))
            except Exception:
                meta[name] = getattr(kv_pool, name)

    if meta_extra:
        meta.update(meta_extra)

    snap = KVSnapshot(
        version=1,
        arch="mha",
        token_ids=token_ids_cpu,
        k=k,
        v=v,
        meta=meta,
    )

    return {
        "version": snap.version,
        "arch": snap.arch,
        "token_ids": snap.token_ids,
        "k": snap.k,
        "v": snap.v,
        "meta": snap.meta,
        "nbytes": snap.nbytes(),
    }


def import_mha_kv_snapshot(
    *,
    kv_pool: Any,
    token_indices: torch.Tensor,
    snapshot: Dict[str, Any],
    strict: bool = True,
) -> None:
    """
    将 snapshot 导入到另一个 MHATokenToKVPool：
    - B 侧通常会先分配 token_indices（T 个槽位）
    - 然后把 snapshot 的 K/V scatter 写回 kv_pool buffers
    """
    if snapshot.get("arch") != "mha":
        raise ValueError(f"Only support arch=mha now, got {snapshot.get('arch')}")
    if strict and int(snapshot.get("version", -1)) != 1:
        raise ValueError(f"Unsupported snapshot version: {snapshot.get('version')}")

    k_cpu: torch.Tensor = snapshot["k"]
    v_cpu: torch.Tensor = snapshot["v"]

    if token_indices.dtype not in (torch.int32, torch.int64):
        token_indices = token_indices.to(torch.int64)

    device = kv_pool.k_buffer[0].device
    idx = token_indices.to(device=device, dtype=torch.int64)

    # 将 CPU snapshot 逐层拷到 device，再 index_copy 写入
    layer_num = int(snapshot["meta"]["layer_num"])
    if strict:
        assert layer_num == int(getattr(kv_pool, "layer_num", len(kv_pool.k_buffer))), \
            f"layer_num mismatch: snap={layer_num} pool={getattr(kv_pool,'layer_num',len(kv_pool.k_buffer))}"

    for l in range(layer_num):
        k_l = k_cpu[l].to(device=device, non_blocking=False).contiguous()
        v_l = v_cpu[l].to(device=device, non_blocking=False).contiguous()
        kv_pool.k_buffer[l].index_copy_(0, idx, k_l)
        kv_pool.v_buffer[l].index_copy_(0, idx, v_l)


def pretty_snapshot_brief(snapshot: Dict[str, Any]) -> str:
    meta = snapshot.get("meta", {})
    return (
        f"KVSnapshot(v={snapshot.get('version')}, arch={snapshot.get('arch')}, "
        f"T={meta.get('token_count')}, L={meta.get('layer_num')}, "
        f"k_per_layer={meta.get('k_shape_per_layer')}, v_per_layer={meta.get('v_shape_per_layer')}, "
        f"dtype={meta.get('dtype')}, nbytes={snapshot.get('nbytes')})"
    )
