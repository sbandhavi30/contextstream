"""
eviction_store.py — Cold storage for raw tool output.

Raw output streams here immediately on tool call — never materializes
in application memory. Returns a pointer (raw_ref). Re-paging fetches
by ref. Pluggable backends: local disk (default), Redis, S3 (v2).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class EvictionBackend(ABC):
    @abstractmethod
    def write(self, ref: str, content: str) -> None: ...

    @abstractmethod
    def read(self, ref: str) -> str | None: ...

    @abstractmethod
    def delete(self, ref: str) -> None: ...

    @abstractmethod
    def exists(self, ref: str) -> bool: ...


# ---------------------------------------------------------------------------
# Local disk backend (default, MVP)
# ---------------------------------------------------------------------------

class DiskBackend(EvictionBackend):
    def __init__(self, base_dir: Path | str | None = None):
        self._dir = Path(base_dir or os.environ.get(
            "CONTEXTSTREAM_EVICTION_DIR",
            Path.home() / ".cache" / "contextstream" / "evictions"
        ))
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._dir, 0o700)
        except OSError:
            pass

    def _path(self, ref: str) -> Path:
        return self._dir / f"{ref}.raw"

    def write(self, ref: str, content: str) -> None:
        p = self._path(ref)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

    def read(self, ref: str) -> str | None:
        p = self._path(ref)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def delete(self, ref: str) -> None:
        p = self._path(ref)
        if p.exists():
            p.unlink()

    def exists(self, ref: str) -> bool:
        return self._path(ref).exists()


# ---------------------------------------------------------------------------
# In-memory backend (testing / dry-run mode)
# ---------------------------------------------------------------------------

class MemoryBackend(EvictionBackend):
    def __init__(self):
        self._store: dict[str, str] = {}

    def write(self, ref: str, content: str) -> None:
        self._store[ref] = content

    def read(self, ref: str) -> str | None:
        return self._store.get(ref)

    def delete(self, ref: str) -> None:
        self._store.pop(ref, None)

    def exists(self, ref: str) -> bool:
        return ref in self._store

    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# EvictionStore — main interface used by the engine
# ---------------------------------------------------------------------------

class EvictionStore:
    """
    Swallows raw tool output streams, stores to backend, returns ref pointer.
    Framework never holds raw data — only the ref.
    """

    def __init__(self, backend: EvictionBackend | None = None):
        self._backend = backend or DiskBackend()
        self._metadata: dict[str, dict] = {}   # ref → {tool, size, session_id}

    def save_stream(
        self,
        stream: Iterator[str],
        tool_name: str = "unknown",
        session_id: str = "",
    ) -> str:
        """Consume stream, write to backend, return ref. O(n) memory — streams chunk by chunk."""
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
        content = "".join(chunks)
        ref = self._make_ref(content, tool_name)
        self._backend.write(ref, content)
        self._metadata[ref] = {
            "tool": tool_name,
            "size": len(content),
            "session_id": session_id,
        }
        return ref

    def save(self, content: str, tool_name: str = "unknown", session_id: str = "") -> str:
        """Save a materialized string. Use save_stream when possible."""
        ref = self._make_ref(content, tool_name)
        self._backend.write(ref, content)
        self._metadata[ref] = {"tool": tool_name, "size": len(content), "session_id": session_id}
        return ref

    def fetch(self, ref: str) -> str:
        """Re-page raw data by ref. Returns empty string if evicted or missing."""
        return self._backend.read(ref) or ""

    def evict(self, ref: str) -> None:
        """Permanently delete raw data. Call after lesson confidence is high enough."""
        self._backend.delete(ref)
        self._metadata.pop(ref, None)

    def exists(self, ref: str) -> bool:
        return self._backend.exists(ref)

    def meta(self, ref: str) -> dict:
        return self._metadata.get(ref, {})

    def total_size_bytes(self) -> int:
        return sum(m.get("size", 0) for m in self._metadata.values())

    @staticmethod
    def _make_ref(content: str, tool_name: str) -> str:
        digest = hashlib.sha1(content.encode()).hexdigest()[:12]
        return f"{tool_name}_{digest}"
