# python/sglang/srt/migration/file_p2p/routes.py
from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from python.sglang.srt.managers.tokenizer_manager import TokenizerManager


class AbandonReqInput(BaseModel):
    rid: str
    dst_http: Optional[str] = None
    bootstrap_room: Optional[int] = None
    reason: Optional[str] = None
    # optional: whether to stop locally after triggering migration
    abort_local: bool = True
    # optional: if you want user to provide a fixed ticket (usually None)
    ticket: Optional[str] = None


class AdoptReqInput(BaseModel):
    ticket: str = Field(..., description="Migration ticket created by source node")
    timeout_s: float = 60.0
    auto_inject: bool = True

    # ---- TP / sharded adopt ----
    # If you run one HTTP process per GPU/TP-rank, you can pass kv_key_suffix explicitly.
    # Recommended: kv_key_suffix=f'tp{tp_rank}'
    kv_key_suffix: Optional[str] = Field(
        default=None,
        description="KV shard suffix, e.g. 'tp0'/'tp1'. None means single-file key.",
    )

    # default True: wait for KV_DONE.<suffix> before reading kv.meta/bin to avoid race
    wait_kv_done: bool = True


def attach_file_p2p_routes(
    app: FastAPI,
    *,
    get_global_state: Callable[[], Optional[Any]],
    node_name: str = "node",
    prefix: str = "/file_p2p",
) -> None:
    """
    注意：这里不要依赖 scheduler 变量。
    在 SGLang 里 scheduler 在 subprocess，HTTP 进程只能通过 tokenizer_manager 发 IPC/RPC。
    所以 route handler 里拿 tokenizer_manager，然后调用你加的新方法即可。
    """
    router = APIRouter(prefix=prefix, tags=["file_p2p"])

    @router.get("/ping")
    async def ping():
        return {"ok": True, "node": node_name}

    @router.post("/abandon_req")
    async def abandon_req(obj: AbandonReqInput, request: Request):
        gs = get_global_state()
        if gs is None:
            raise HTTPException(
                status_code=503, detail="server not ready (_global_state is None)"
            )

        tm: TokenizerManager = gs.tokenizer_manager
        if not hasattr(tm, "abandon_req"):
            raise HTTPException(
                status_code=501,
                detail="tokenizer_manager.abandon_req is not implemented yet",
            )

        # NOTE: tm.abandon_req should send AbandonReq IPC to scheduler and await AbandonReqOutput.
        # Here we just forward params. Keep it async-friendly.
        ret = await tm.abandon_req(
            rid=obj.rid,
            dst_http=obj.dst_http,
            bootstrap_room=obj.bootstrap_room,
            reason=obj.reason,
            abort_local=obj.abort_local,
        )
        return {"ok": True, "node": node_name, "result": ret}

    @router.post("/adopt_req")
    async def adopt_req(obj: AdoptReqInput, request: Request):
        gs = get_global_state()
        if gs is None:
            raise HTTPException(
                status_code=503, detail="server not ready (_global_state is None)"
            )

        tm: TokenizerManager = gs.tokenizer_manager
        if not hasattr(tm, "adopt_req"):
            raise HTTPException(
                status_code=501,
                detail="tokenizer_manager.adopt_req is not implemented yet",
            )

        # NOTE: tm.adopt_req can either:
        #   - call local scheduler RPC to run engine.adopt_req_from_store(...)
        #   - or do it in tokenizer process if it has access to FileKVStore (not recommended)
        ret = await tm.adopt_req(
            ticket=obj.ticket,
            timeout_s=obj.timeout_s,
            auto_inject=obj.auto_inject,
            kv_key_suffix=obj.kv_key_suffix,
            wait_kv_done=obj.wait_kv_done,
        )
        return {"ok": True, "node": node_name, "result": ret}

    app.include_router(router)
