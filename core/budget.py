"""
budget.py — Token budget tracker and pre-flight overflow prevention.

Counts tokens in the active context window per model using tiktoken.
Falls back to char/4 estimate if tiktoken unavailable.
Fires pressure events at configurable thresholds (0.7, 0.9, 0.95).
Pre-flight check runs BEFORE each tool call — never overflows mid-generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

# Model context limits (tokens)
MODEL_LIMITS: dict[str, int] = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-opus-4-7":           200_000,
    "gpt-4o":                    128_000,
    "gpt-4o-mini":               128_000,
}
DEFAULT_LIMIT = 200_000

# Tiktoken encoding aliases (model → tiktoken encoding name)
TIKTOKEN_ENCODINGS: dict[str, str] = {
    "gpt-4o":      "o200k_base",
    "gpt-4o-mini": "o200k_base",
}


class PressureLevel(str, Enum):
    NORMAL   = "normal"    # < 70%
    WARN     = "warn"      # 70–89%
    HIGH     = "high"      # 90–94%
    CRITICAL = "critical"  # >= 95%


@dataclass
class BudgetStatus:
    used_tokens:   int
    limit_tokens:  int
    pressure:      PressureLevel
    headroom:      int          # tokens remaining

    @property
    def usage_fraction(self) -> float:
        return self.used_tokens / self.limit_tokens if self.limit_tokens else 0.0


class BudgetTracker:
    """
    Tracks token usage for a single agent session.
    Call update() after every ledger append or context change.
    Call preflight() before every tool call.
    """

    THRESHOLDS = {
        PressureLevel.CRITICAL: 0.95,
        PressureLevel.HIGH:     0.90,
        PressureLevel.WARN:     0.70,
    }

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        overhead_tokens: int = 1_000,    # system prompt + scaffolding estimate
    ):
        self.model          = model
        self.limit          = MODEL_LIMITS.get(model, DEFAULT_LIMIT)
        self.overhead       = overhead_tokens
        self._used          = overhead_tokens
        self._callbacks: dict[PressureLevel, list[Callable]] = {
            p: [] for p in PressureLevel
        }
        self._encoder = self._load_encoder(model)

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        """Count tokens in a string."""
        if self._encoder:
            return len(self._encoder.encode(text))
        return max(1, len(text) // 4)   # char/4 fallback

    def update(self, text: str) -> BudgetStatus:
        """Add tokens for newly appended text. Returns current status."""
        self._used += self.count(text)
        status = self._status()
        self._fire_callbacks(status.pressure)
        return status

    def preflight(self, estimated_output_tokens: int = 2_000) -> BudgetStatus:
        """
        Check if there is headroom for the next tool call + expected output.
        Caller should abort tool call if pressure == CRITICAL.
        """
        projected = self._used + estimated_output_tokens
        pressure  = self._classify(projected / self.limit)
        return BudgetStatus(
            used_tokens=projected,
            limit_tokens=self.limit,
            pressure=pressure,
            headroom=self.limit - projected,
        )

    def set_used(self, tokens: int) -> None:
        """Directly set usage (e.g. after rendering full prompt)."""
        self._used = tokens

    def reset(self) -> None:
        self._used = self.overhead

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_pressure(self, level: PressureLevel, callback: Callable) -> None:
        """Register callback for a pressure threshold crossing."""
        self._callbacks[level].append(callback)

    def _fire_callbacks(self, level: PressureLevel) -> None:
        for cb in self._callbacks.get(level, []):
            try:
                cb(self._status())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _status(self) -> BudgetStatus:
        pressure = self._classify(self._used / self.limit)
        return BudgetStatus(
            used_tokens=self._used,
            limit_tokens=self.limit,
            pressure=pressure,
            headroom=self.limit - self._used,
        )

    @staticmethod
    def _classify(fraction: float) -> PressureLevel:
        if fraction >= BudgetTracker.THRESHOLDS[PressureLevel.CRITICAL]:
            return PressureLevel.CRITICAL
        if fraction >= BudgetTracker.THRESHOLDS[PressureLevel.HIGH]:
            return PressureLevel.HIGH
        if fraction >= BudgetTracker.THRESHOLDS[PressureLevel.WARN]:
            return PressureLevel.WARN
        return PressureLevel.NORMAL

    @staticmethod
    def _load_encoder(model: str):
        try:
            import tiktoken
            enc_name = TIKTOKEN_ENCODINGS.get(model, "cl100k_base")
            return tiktoken.get_encoding(enc_name)
        except (ImportError, Exception):
            return None   # fall back to char/4
