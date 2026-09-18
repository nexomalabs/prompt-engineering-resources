"""Recorded model responses, so labs can be tested without an API key.

LAB_PROJECT_STANDARD.md §6 defines three CI tiers. Tier 2 runs every lab's tests
on every push with no key and no cost, which is only possible if model responses
can be replayed from disk.

    NEXOMA_LAB_MODE=fixture   replay from fixtures/ (default in CI)
    NEXOMA_LAB_MODE=record    call the real API and save responses
    NEXOMA_LAB_MODE=live      call the real API, save nothing
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib


def lab_mode() -> str:
    mode = os.environ.get("NEXOMA_LAB_MODE", "fixture").lower()
    if mode not in {"fixture", "record", "live"}:
        raise ValueError(f"NEXOMA_LAB_MODE must be fixture, record or live (got {mode!r})")
    return mode


class FixtureMissing(RuntimeError):
    """Raised when fixture mode is active but no recording exists."""


class FixtureStore:
    """Content-addressed store of model responses.

    The key is a hash of the request, so a changed prompt misses the cache and
    fails loudly rather than silently replaying a stale answer.
    """

    def __init__(self, directory: str | pathlib.Path):
        self.dir = pathlib.Path(directory)

    @staticmethod
    def key(request: dict) -> str:
        blob = json.dumps(request, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def path(self, request: dict) -> pathlib.Path:
        return self.dir / f"{self.key(request)}.json"

    def load(self, request: dict) -> dict:
        p = self.path(request)
        if not p.exists():
            raise FixtureMissing(
                f"no fixture for this request at {p}.\n"
                f"Record one with:  NEXOMA_LAB_MODE=record uv run python solution/lab.py"
            )
        return json.loads(p.read_text(encoding="utf-8"))["response"]

    def save(self, request: dict, response: dict) -> pathlib.Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(request)
        p.write_text(
            json.dumps({"request": request, "response": response}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return p

    def call(self, request: dict, live_fn):
        """Replay, record, or pass through, according to NEXOMA_LAB_MODE."""
        mode = lab_mode()
        if mode == "fixture":
            return self.load(request)
        response = live_fn(request)
        if mode == "record":
            self.save(request, response)
        return response
