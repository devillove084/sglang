from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

@dataclass
class FileKVStore:
    """
    极简 KVStore（单机 / NFS 共享目录做 rendezvous + 数据落盘模拟）：
    - put/get: torch.save/torch.load 存 meta / 小对象
    - put_bytes: 原子写二进制 blob（tmp + rename），适合存 KV raw bytes
    - wait/exists: 轮询等待 key 出现
    """
    root_dir: Path
    suffix: str = ".pt"

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_of(self, key: str, suffix: Optional[str] = None) -> Path:
        safe_key = key.replace("/", "_")
        suf = self.suffix if suffix is None else suffix
        return self.root_dir / f"{safe_key}{suf}"

    # -------- meta (torch.save) --------
    def put(self, key: str, obj: Dict[str, Any]) -> Path:
        path = self._path_of(key, self.suffix)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic rename on POSIX
        return path

    def get(self, key: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
        path = self._path_of(key, self.suffix)
        return torch.load(path, map_location=map_location)

    def exists(self, key: str, suffix: Optional[str] = None) -> bool:
        return self._path_of(key, suffix).exists()

    def wait(self, key: str, timeout_s: float = 60.0, poll_s: float = 0.05, suffix: Optional[str] = None) -> Path:
        t0 = time.time()
        path = self._path_of(key, suffix)
        while time.time() - t0 < timeout_s:
            if path.exists():
                return path
            time.sleep(poll_s)
        raise TimeoutError(f"Timeout waiting key={key} at {path}")

    # -------- raw blob --------
    def put_bytes(self, key: str, data: bytes, suffix: str = ".bin") -> Path:
        path = self._path_of(key, suffix)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path

    def get_bytes_path(self, key: str, suffix: str = ".bin") -> Path:
        return self._path_of(key, suffix)

    def wait_bytes_path(self, key: str, timeout_s: float = 60.0, poll_s: float = 0.05, suffix: str = ".bin") -> Path:
        return self.wait(key, timeout_s=timeout_s, poll_s=poll_s, suffix=suffix)
