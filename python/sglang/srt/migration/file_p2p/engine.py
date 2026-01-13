# python/sglang/srt/migration/file_p2p/engine.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from sglang.srt.kv_store.file_kv_store import FileKVStore
from sglang.srt.kv_store.kv_snapshot_page_blob import (
    export_kv_pages_to_files,
    import_kv_pages_from_files,
)
from sglang.srt.disaggregation.utils import kv_to_page_indices


@dataclass
class MigrationTicket:
    ticket: str
    rid: str
    created_at: float


class MigrationBusyError(RuntimeError):
    pass


class MigrationNotFoundError(RuntimeError):
    pass

_FILE_P2P_PREFIX = "/file_p2p"

def _normalize_peer_base(peer_url: str) -> str:
    """
    Accept:
      - http://host:port
      - http://host:port/
      - http://host:port/file_p2p
      - http://host:port/file_p2p/
    Return:
      - http://host:port/file_p2p
    """
    u = (peer_url or "").strip()
    if not u:
        return u
    u = u.rstrip("/")
    if u.endswith(_FILE_P2P_PREFIX):
        return u
    return u + _FILE_P2P_PREFIX

def _safe_sampling_params(sp: Any) -> Optional[Dict[str, Any]]:
    if sp is None:
        return None
    if isinstance(sp, dict):
        return sp
    if hasattr(sp, "to_dict"):
        try:
            d = sp.to_dict()
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    # 最保守：不要把自定义对象塞进 req_state
    return None


class FileP2PMigrationEngine:
    """
    P2P Req migration engine (File backend).

    A 侧（source）：
      - freeze req（scheduler 层做 safe-point 更好）
      - dump req_state + kv pages to FileKVStore
      - per-shard KV_DONE marker
      - leader 写 DONE marker + notify B (/adopt_req)
      - optionally finalize (remove req + free slots) on all ranks

    B 侧（dest）：
      - wait DONE
      - wait KV_DONE.<suffix> (per rank)
      - load req_state + kv shard
      - alloc req slots + token slots
      - import kv
      - write req_to_token mapping
      - inject req into scheduler
    """

    def __init__(
        self,
        *,
        scheduler: Any,
        store: FileKVStore,
        node_name: str,
    ):
        self.scheduler = scheduler
        self.store = store
        self.node_name = node_name

    # --------------------------
    # helpers
    # --------------------------

    @staticmethod
    def _kv_key(key_prefix: str, kv_key_suffix: Optional[str]) -> str:
        # single-rank: mig/{ticket}/kv
        # tp shard:    mig/{ticket}/kv.tp0  / kv.tp1 ...
        if kv_key_suffix is None:
            return f"{key_prefix}/kv"
        return f"{key_prefix}/kv.{kv_key_suffix}"

    @staticmethod
    def _kv_done_key(key_prefix: str, kv_key_suffix: Optional[str]) -> str:
        # per-shard completion marker (avoid race)
        if kv_key_suffix is None:
            return f"{key_prefix}/KV_DONE"
        return f"{key_prefix}/KV_DONE.{kv_key_suffix}"

    def _notify_peer_adopt(
        self, ticket: str, peer_url: str, timeout_s: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Notify dest node to adopt. peer_url can be base or already include /file_p2p.
        """
        peer_base = _normalize_peer_base(peer_url)
        try:
            r = requests.post(
                f"{peer_base}/adopt_req",
                json={"ticket": ticket, "timeout_s": timeout_s},
                timeout=(2, 10),
            )
            ok = r.status_code // 100 == 2
            if not ok:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            return True, None
        except Exception as e:
            return False, repr(e)

    # --------------------------
    # public APIs
    # --------------------------

    def abandon_req_to_peer(
        self,
        *,
        rid: str,
        peer_url: str,
        timeout_s: float = 60.0,
        # ---- TP/sharded controls ----
        ticket: Optional[str] = None,
        kv_key_suffix: Optional[str] = None,          # e.g. "tp0" / "tp1"
        write_req_state: bool = True,                 # leader=True, others=False
        write_done: bool = True,                      # leader=True, others=False
        notify: bool = True,                          # leader=True, others=False
        finalize: bool = False,                        # True on all ranks
        allow_running: bool = False,                  # scheduler safe-point 时可 True
        wait_kv_done_suffixes: Optional[List[str]] = None,  # leader 可等待所有 shard done
        poll_s: float = 0.05,
    ) -> Dict[str, Any]:
        """
        A side: called by scheduler upon AbandonReq.

        兼容旧调用：不传新参数时行为与旧版一致（写 req_state/kv/DONE/notify/finalize）。

        TP 推荐：
          - 所有 rank 传同 ticket
          - kv_key_suffix=f"tp{tp_rank}"
          - non-leader: write_req_state=False, write_done=False, notify=False
          - leader:     write_req_state=True,  write_done=True,  notify=True,
                        wait_kv_done_suffixes=[f"tp{i}" for i in range(tp_size)]
          - finalize=True on all ranks
          - allow_running=True（scheduler 已做 safe-point/barrier）
        """
        if ticket is None:
            ticket = f"{rid}-{uuid.uuid4().hex}"
        key_prefix = f"mig/{ticket}"

        # 1) extract snapshot
        req, snap = self._extract_req_snapshot_for_migration(
            rid, allow_running=allow_running
        )

        # 2) write req_state (leader-only recommended)
        if write_req_state:
            self.store.put(f"{key_prefix}/req_state", snap["req_state"])

        # 3) write kv snapshot (per-rank shard)
        kv_key = self._kv_key(key_prefix, kv_key_suffix)
        export_kv_pages_to_files(
            kv_pool=snap["kv_pool"],
            page_indices=snap["page_indices"],
            store=self.store,
            key=kv_key,
            token_ids=snap.get("fill_ids_cpu"),
            meta_extra={
                "rid": rid,
                "seq_len": snap["seq_len"],
                "from_node": self.node_name,
                "kv_key_suffix": kv_key_suffix,
            },
        )

        # 3.1) mark kv shard done
        kv_done_key = self._kv_done_key(key_prefix, kv_key_suffix)
        self.store.put(
            kv_done_key,
            {
                "t": time.time(),
                "rid": rid,
                "from_node": self.node_name,
                "kv_key_suffix": kv_key_suffix,
                "kv_key": kv_key,
            },
        )

        # 4) leader waits all shards (optional), then write DONE
        if write_done:
            if wait_kv_done_suffixes:
                for suf in wait_kv_done_suffixes:
                    self.store.wait(
                        self._kv_done_key(key_prefix, suf),
                        timeout_s=timeout_s,
                        poll_s=poll_s,
                    )

            self.store.put(
                f"{key_prefix}/DONE",
                {
                    "t": time.time(),
                    "rid": rid,
                    "from_node": self.node_name,
                    "ticket": ticket,
                    "kv_done_suffixes": wait_kv_done_suffixes,
                },
            )

        # 5) leader notify B
        notify_ok = False
        notify_err = None
        peer_base = _normalize_peer_base(peer_url)
        if notify:
            notify_ok, notify_err = self._notify_peer_adopt(ticket, peer_base, timeout_s)

        # 6) finalize (all ranks)
        if finalize:
            self._finalize_abandon(req)

        return {
            "ticket": ticket,
            "rid": rid,
            "store_prefix": key_prefix,
            "peer_url": peer_base,
            "kv_key": kv_key,
            "kv_key_suffix": kv_key_suffix,
            "kv_done_key": kv_done_key,
            "notify_ok": notify_ok,
            "notify_err": notify_err,
            "req_obj": req,
        }

    def adopt_req_from_store(
        self,
        *,
        ticket: str,
        timeout_s: float = 60.0,
        auto_inject: bool = True,
        # ---- TP/sharded controls ----
        kv_key_suffix: Optional[str] = None,   # e.g. "tp0" / "tp1"
        poll_s: float = 0.05,
        wait_kv_done: bool = True,             # 每 rank 等待自己 shard KV_DONE
    ) -> Dict[str, Any]:
        """
        B side: called by /adopt_req (or scheduler control plane).

        单卡：
          kv_key_suffix=None -> 读 mig/{ticket}/kv.meta + kv.bin

        TP：
          每个 rank 传 kv_key_suffix=f"tp{tp_rank}"
          会先 wait(KV_DONE.<suffix>) 再读 meta/bin，避免 race。
        """
        key_prefix = f"mig/{ticket}"

        # 1) wait DONE (leader on A should write it after all kv shards done)
        self.store.wait(f"{key_prefix}/DONE", timeout_s=timeout_s, poll_s=poll_s)

        # 2) wait my shard done (optional but strongly recommended)
        if wait_kv_done:
            self.store.wait(
                self._kv_done_key(key_prefix, kv_key_suffix),
                timeout_s=timeout_s,
                poll_s=poll_s,
            )

        # 3) load req_state + kv_meta (per shard)
        req_state = self.store.get(f"{key_prefix}/req_state", map_location="cpu")

        kv_key = self._kv_key(key_prefix, kv_key_suffix)
        kv_meta = self.store.get(f"{kv_key}.meta", map_location="cpu")

        rid = str(req_state["rid"])
        seq_len = int(req_state["seq_len"])

        # 4) allocate slots on B
        alloc = self._alloc_slots_for_adopt(seq_len=seq_len)
        # dst_token_indices = alloc["token_indices"]  # Tensor [seq_len] on device
        try:
            dst_token_indices = alloc["token_indices"]  # CPU tensor
            dst_page_indices = kv_to_page_indices(
                dst_token_indices.detach().cpu().numpy(),
                page_size=int(kv_meta["page_size"]),
            )

            # 5) import kv pages for my shard
            import_kv_pages_from_files(
                kv_pool=alloc["kv_pool"],
                dst_page_indices=dst_page_indices,
                store=self.store,
                key=kv_key,
                strict=True,
            )

            # 6) write req_to_token mapping
            self._write_req_to_token_mapping(
                req_pool_idx=alloc["req_pool_idx"],
                token_indices=dst_token_indices,
                seq_len=seq_len,
            )

            # 7) inject req into scheduler
            req_obj = None
            if auto_inject:
                req_obj = self._inject_adopted_req(
                    req_state=req_state, req_pool_idx=alloc["req_pool_idx"]
                )
        except Exception:
            self._rollback_adopt_alloc(
                req_pool_idx=int(alloc["req_pool_idx"]),
                token_indices=alloc["token_indices"],
            )
            raise

        return {
            "ticket": ticket,
            "rid": rid,
            "seq_len": seq_len,
            "req_pool_idx": int(alloc["req_pool_idx"]),
            "auto_inject": auto_inject,
            "injected": req_obj is not None,
            "kv_key": kv_key,
            "kv_key_suffix": kv_key_suffix,
            "kv_meta_brief": {
                "arch": kv_meta.get("arch"),
                "page_size": kv_meta.get("page_size"),
                "layer_num": kv_meta.get("layer_num"),
            },
        }

    def _notify_peer_adopt(
        self, ticket: str, peer_url: str, timeout_s: float
    ) -> Tuple[bool, Optional[str]]:
        try:
            r = requests.post(
                f"{peer_url}/adopt_req",
                json={"ticket": ticket, "timeout_s": timeout_s},
                timeout=(2, 10),
            )
            ok = r.status_code // 100 == 2
            if not ok:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            return True, None
        except Exception as e:
            return False, repr(e)

    # --------------------------
    # A side internals
    # --------------------------

    def _allocator_protect_tokens(self, allocator: Any, token_indices: torch.Tensor) -> None:
        """
        Important: token_to_kv_pool_allocator 通常要求 alloc() 后把 tokens 计入 protected/active，
        否则 runtime checker 会认为 leaked。
        这里用反射兼容不同版本方法名。
        """
        cand = [
            "protect",
            "protect_tokens",
            "pin",
            "pin_tokens",
            "mark_protected",
            "register_allocated",
            "on_allocated",
            "acquire",
        ]
        last_err = None
        for name in cand:
            fn = getattr(allocator, name, None)
            if fn is None:
                continue
            try:
                fn(token_indices)
                return
            except Exception as e:
                last_err = e

        # 如果没有任何已知 API，宁可 fail-fast（并让上层 rollback）也不要继续跑，
        # 否则 scheduler idle 自检必崩。
        raise RuntimeError(
            f"Cannot protect adopted tokens: allocator has no known protect/pin API. "
            f"Tried={cand}. last_err={last_err!r}. "
            f"Please hook _allocator_protect_tokens() for your allocator implementation."
        )


    def _rollback_adopt_alloc(self, *, req_pool_idx: int, token_indices: torch.Tensor) -> None:
        """
        adopt 失败时回滚：free tokens + free req slot
        """
        allocator = self._get_token_to_kv_pool_allocator()
        req_to_token_pool = self._get_req_to_token_pool()

        # 一些 allocator 需要先 unprotect，再 free（按版本兼容）
        for name in ("unprotect", "unpin", "unprotect_tokens", "unpin_tokens", "release"):
            fn = getattr(allocator, name, None)
            if fn is None:
                continue
            try:
                fn(token_indices)
                break
            except Exception:
                pass

        if isinstance(token_indices, torch.Tensor) and token_indices.is_cuda:
            token_indices = token_indices.cpu()
        if hasattr(allocator, "free"):
            allocator.free(token_indices)
        elif hasattr(allocator, "free_tokens"):
            allocator.free_tokens(token_indices)

        try:
            req_to_token_pool.free(req_pool_idx)
        except Exception:
            pass


    def _extract_req_snapshot_for_migration(
        self, rid: str, allow_running: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        从 scheduler 抓到 req + kv indices + fill_ids 等，并且要保证此刻 KV 内容稳定。

        - allow_running=False: 旧行为：在 running_batch 就认为 busy
        - allow_running=True: 允许抓 running req（scheduler 需先做 safe-point 同步）
        """
        req = self._find_req_by_rid(rid)
        if req is None:
            raise MigrationNotFoundError(
                f"rid {rid} not found on node {self.node_name}"
            )

        if (not allow_running) and self._is_req_busy(req):
            raise MigrationBusyError(
                f"rid {rid} is busy (in running batch), retry later"
            )

        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is None:
            origin = getattr(req, "origin_input_ids", [])
            out = getattr(req, "output_ids", [])
            fill_ids = list(origin) + list(out)

        seq_len = len(fill_ids)
        if seq_len <= 0:
            raise RuntimeError(f"rid {rid}: seq_len=0 cannot migrate")

        req_pool_idx = int(getattr(req, "req_pool_idx"))

        req_to_token_pool = self._get_req_to_token_pool()
        token_indices = req_to_token_pool.req_to_token[req_pool_idx, :seq_len].detach()

        page_size = int(self._get_kv_pool().page_size)
        page_indices = kv_to_page_indices(
            token_indices.cpu().numpy(), page_size=page_size
        )

        sp = getattr(req, "sampling_params", None)
        sp_dict = _safe_sampling_params(sp)
        req_state = {
            "rid": rid,
            "seq_len": seq_len,
            "fill_ids": torch.tensor(fill_ids, dtype=torch.int32),
            "output_ids": torch.tensor(
                getattr(req, "output_ids", []), dtype=torch.int32
            ),
            "sampling_params": sp_dict,
            "return_logprob": bool(getattr(req, "return_logprob", False)),
            "created_time": float(getattr(req, "created_time", time.time())),
        }

        snap = {
            "rid": rid,
            "seq_len": seq_len,
            "req_pool_idx": req_pool_idx,
            "token_indices": token_indices,
            "page_indices": page_indices,
            "kv_pool": self._get_kv_pool(),
            "fill_ids_cpu": req_state["fill_ids"],
            "req_state": req_state,
        }
        return req, snap

    def _finalize_abandon(self, req: Any) -> None:
        """
        A 侧在落盘完成后，把 req 从本地“彻底移除”并释放 kv slots
        """
        self._remove_req_from_scheduler(req)
        self._free_req_slots(req)

    # --------------------------
    # B side internals
    # --------------------------

    # def _alloc_slots_for_adopt(self, seq_len: int) -> Dict[str, Any]:
    #     """
    #     B 侧为迁移进来的 req 分配：
    #       - req_pool_idx
    #       - token_indices（长度 seq_len）
    #     """
    #     req_to_token_pool = self._get_req_to_token_pool()
    #     req_slots = req_to_token_pool.alloc(1)
    #     if req_slots is None:
    #         raise RuntimeError("No free req slots on B")
    #     req_pool_idx = int(req_slots[0])

    #     allocator = self._get_token_to_kv_pool_allocator()
    #     token_indices = allocator.alloc(seq_len)
    #     if token_indices is None:
    #         raise RuntimeError(f"No free token slots on B for seq_len={seq_len}")

    #     if not isinstance(token_indices, torch.Tensor):
    #         token_indices = torch.tensor(
    #             token_indices,
    #             dtype=torch.int64,
    #             device=self._get_kv_pool().k_buffer[0].device,
    #         )

    #     return {
    #         "req_pool_idx": req_pool_idx,
    #         "token_indices": token_indices.to(torch.int64),
    #         "kv_pool": self._get_kv_pool(),
    #     }

    def _allocator_sizes(self, allocator: Any) -> Tuple[int, int, int]:
        def _get(name: str) -> int:
            v = getattr(allocator, name)
            return int(v() if callable(v) else v)
        return _get("available_size"), _get("protected_size"), _get("evictable_size")

    def _allocator_try_call(self, fn, tok, want_n: int) -> bool:
        """
        兼容不同签名：fn(tok) / fn(tok, n) / fn(tok, True)
        """
        try:
            fn(tok)
            return True
        except TypeError:
            pass
        # 尝试 2 参数
        try:
            fn(tok, want_n)
            return True
        except TypeError:
            pass
        try:
            fn(tok, True)
            return True
        except TypeError:
            return False

    def _allocator_protect_tokens(self, allocator: Any, token_indices: Any) -> None:
        cand = [
            # 先放最可能的
            "protect_tokens",
            "protect",
            "pin_tokens",
            "pin",
            "mark_protected",
            "register_allocated",
            "on_allocated",
            "acquire",
        ]

        avail0, prot0, ev0 = self._allocator_sizes(allocator)

        # 构造多种 token 表示：CPU tensor / list / numpy
        toks = []
        if isinstance(token_indices, torch.Tensor):
            ti = token_indices.detach()
            if ti.is_cuda:
                toks.append(ti.cpu())
            else:
                toks.append(ti)
            toks.append(ti.cpu().to(torch.int64))
            toks.append(ti.cpu().to(torch.int64).tolist())
        else:
            # list / np
            toks.append(token_indices)
            try:
                toks.append(torch.tensor(token_indices, dtype=torch.int64, device="cpu"))
            except Exception:
                pass
            try:
                toks.append(list(token_indices))
            except Exception:
                pass

        want_n = len(token_indices) if not isinstance(token_indices, torch.Tensor) else int(token_indices.numel())

        last_err = None
        for name in cand:
            fn = getattr(allocator, name, None)
            if fn is None:
                continue

            for tok in toks:
                try:
                    ok = self._allocator_try_call(fn, tok, want_n)
                    if not ok:
                        continue

                    # ✅ 关键：验证 protected_size 真的变大了，否则继续尝试其他方法/形态
                    avail1, prot1, ev1 = self._allocator_sizes(allocator)
                    if prot1 > prot0:
                        return

                except Exception as e:
                    last_err = e

        avail1, prot1, ev1 = self._allocator_sizes(allocator)
        raise RuntimeError(
            f"Cannot protect adopted tokens: protected_size not increased. "
            f"before(avail,prot,ev)=({avail0},{prot0},{ev0}) after=({avail1},{prot1},{ev1}). "
            f"Tried methods={cand}. last_err={last_err!r}."
        )

    def _alloc_slots_for_adopt(self, seq_len: int) -> Dict[str, Any]:
        req_to_token_pool = self._get_req_to_token_pool()
        req_slots = req_to_token_pool.alloc(1)
        if req_slots is None:
            raise RuntimeError("No free req slots on B")
        req_pool_idx = int(req_slots[0])

        allocator = self._get_token_to_kv_pool_allocator()

        token_indices = allocator.alloc(seq_len)
        if token_indices is None:
            try:
                req_to_token_pool.free(req_pool_idx)
            except Exception:
                pass
            raise RuntimeError(f"No free token slots on B for seq_len={seq_len}")

        # ✅ 关键：不要把 token_indices 搬到 GPU
        if isinstance(token_indices, torch.Tensor):
            token_indices_cpu = token_indices.detach()
            if token_indices_cpu.is_cuda:
                token_indices_cpu = token_indices_cpu.cpu()
            token_indices_cpu = token_indices_cpu.to(torch.int64)
        else:
            token_indices_cpu = torch.tensor(token_indices, dtype=torch.int64, device="cpu")

        # ✅ 关键：protect + 验证 protected_size 增长
        self._allocator_protect_tokens(allocator, token_indices_cpu)

        return {
            "req_pool_idx": req_pool_idx,
            "token_indices": token_indices_cpu,  # CPU tensor
            "kv_pool": self._get_kv_pool(),
        }


    def _write_req_to_token_mapping(
        self, req_pool_idx: int, token_indices: torch.Tensor, seq_len: int
    ) -> None:
        req_to_token_pool = self._get_req_to_token_pool()
        req_to_token_pool.write(
            (req_pool_idx, slice(0, seq_len)),
            token_indices.detach().to(req_to_token_pool.req_to_token.device),
        )

    def _inject_adopted_req(
        self, req_state: Dict[str, Any], req_pool_idx: int
    ) -> Optional[Any]:
        """
        把迁移进来的 req 恢复成 Scheduler 能继续 decode 的对象。
        需要按你本地版本对齐：Req 构造字段/标志位/入队方式。
        """
        from sglang.srt.managers.schedule_batch import Req  # type: ignore

        rid = str(req_state["rid"])
        fill_ids = req_state["fill_ids"].tolist()
        output_ids = req_state["output_ids"].tolist()

        req = Req(
            rid=rid,
            origin_input_ids=fill_ids,
        )
        req.req_pool_idx = req_pool_idx
        req.fill_ids = fill_ids
        req.output_ids = output_ids
        req.return_logprob = bool(req_state.get("return_logprob", False))

        # sp = req_state.get("sampling_params", None)
        # if sp is not None:
        #     try:
        #         req.sampling_params = req.sampling_params.from_dict(sp)  # type: ignore
        #     except Exception:
        #         req.sampling_params = sp
        sp = req_state.get("sampling_params", None)
        if isinstance(sp, dict) and sp:
            try:
                from sglang.srt.sampling.sampling_params import SamplingParams  # 具体路径按你代码库为准
                if hasattr(SamplingParams, "from_dict"):
                    req.sampling_params = SamplingParams.from_dict(sp)  # type: ignore
                else:
                    req.sampling_params = SamplingParams(**sp)  # type: ignore
            except Exception:
                # fallback：起码别崩
                req.sampling_params = sp

        # 标记“prefill 已完成”
        if hasattr(req, "prefilled"):
            req.prefilled = True
        if hasattr(req, "is_prefill_done"):
            req.is_prefill_done = True

        # 注入：最简单先扔 waiting_queue（你之后可以做更精细的 decode-ready 入队）
        sched = self.scheduler
        if hasattr(sched, "waiting_queue"):
            sched.waiting_queue.append(req)
            return req

        raise RuntimeError(
            "Cannot inject req: scheduler has no waiting_queue; please hook _inject_adopted_req"
        )

    # --------------------------
    # Scheduler hooks (version dependent)
    # --------------------------

    def _find_req_by_rid(self, rid: str) -> Optional[Any]:
        s = self.scheduler

        for name in ("rid_to_req", "req_table", "requests", "request_table"):
            tbl = getattr(s, name, None)
            if isinstance(tbl, dict) and rid in tbl:
                return tbl[rid]

        for qname in (
            "waiting_queue",
            "disagg_prefill_inflight_queue",
            "inflight_queue",
        ):
            q = getattr(s, qname, None)
            if q is None:
                continue
            try:
                for req in q:
                    if getattr(req, "rid", None) == rid:
                        return req
            except Exception:
                pass

        rb = getattr(s, "running_batch", None)
        if rb is not None and hasattr(rb, "reqs"):
            for req in rb.reqs:
                if getattr(req, "rid", None) == rid:
                    return req

        return None

    def _is_req_busy(self, req: Any) -> bool:
        rb = getattr(self.scheduler, "running_batch", None)
        if rb is not None and hasattr(rb, "reqs"):
            for r in rb.reqs:
                if r is req:
                    return True
        return False

    def _remove_req_from_scheduler(self, req: Any) -> None:
        s = self.scheduler
        for qname in (
            "waiting_queue",
            "disagg_prefill_inflight_queue",
            "inflight_queue",
        ):
            q = getattr(s, qname, None)
            if q is None:
                continue
            try:
                if req in q:
                    q.remove(req)
            except Exception:
                pass

        for name in ("rid_to_req", "req_table", "requests", "request_table"):
            tbl = getattr(s, name, None)
            if isinstance(tbl, dict):
                rid = getattr(req, "rid", None)
                if rid in tbl and tbl[rid] is req:
                    del tbl[rid]

    def _free_req_slots(self, req: Any) -> None:
        req_pool_idx = int(getattr(req, "req_pool_idx"))
        seq_len = (
            len(getattr(req, "fill_ids", []))
            or len(getattr(req, "origin_input_ids", []))
        )

        req_to_token_pool = self._get_req_to_token_pool()
        token_indices = req_to_token_pool.req_to_token[req_pool_idx, :seq_len].detach()

        if isinstance(token_indices, torch.Tensor) and token_indices.is_cuda:
            oken_indices = token_indices.cpu()

        allocator = self._get_token_to_kv_pool_allocator()
        if hasattr(allocator, "free"):
            allocator.free(token_indices)
        elif hasattr(allocator, "free_tokens"):
            allocator.free_tokens(token_indices)

        req_to_token_pool.free(req_pool_idx)

    def _get_req_to_token_pool(self) -> Any:
        p = getattr(self.scheduler, "req_to_token_pool", None)
        if p is None:
            raise RuntimeError("scheduler.req_to_token_pool not found")
        return p

    def _get_token_to_kv_pool_allocator(self) -> Any:
        a = getattr(self.scheduler, "token_to_kv_pool_allocator", None)
        if a is None:
            mr = getattr(self.scheduler, "tp_worker", None)
            if mr is not None:
                mr = getattr(mr, "model_runner", None)
            a = getattr(mr, "token_to_kv_pool_allocator", None)
        if a is None:
            raise RuntimeError("token_to_kv_pool_allocator not found")
        return a

    def _get_kv_pool(self) -> Any:
        allocator = self._get_token_to_kv_pool_allocator()
        if hasattr(allocator, "get_kvcache"):
            return allocator.get_kvcache()
        p = getattr(self.scheduler, "token_to_kv_pool", None)
        if p is None:
            raise RuntimeError("kv_pool not found")
        return p
