"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Design choices:
- Logs are themselves a leak vector, so text is stored *redacted*
  (secrets + PII) and the raw input is kept only as a SHA-256 digest —
  enough to prove what was sent without re-exposing it.
- Entries are hash-chained (``prev_hash`` -> ``entry_hash``) so any edit or
  deletion after the fact breaks ``verify_chain()``.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter, detect_secret_leak

_GENESIS_HASH = "0" * 64
_MAX_STORED_CHARS = 500


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _sanitize(text: str) -> str:
    """Redact secrets/PII before anything is written to the log."""
    if detect_secret_leak(text):
        return "[REDACTED: protected secret]"
    redacted = content_filter(text or "")["redacted"]
    if len(redacted) > _MAX_STORED_CHARS:
        redacted = redacted[:_MAX_STORED_CHARS] + f"… [+{len(redacted) - _MAX_STORED_CHARS} chars]"
    return redacted


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id. Returns the request_id."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = time.perf_counter()
        self._pending[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": _sanitize(text),
            "input_sha256": _sha256(text),
            "input_chars": len(text or ""),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        reason: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        started = self._open.pop(request_id, None) if request_id else None
        entry = self._pending.pop(request_id, None) if request_id else None
        if entry is None:
            entry = {
                "request_id": request_id or uuid.uuid4().hex,
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": None,
                "input_sha256": None,
                "input_chars": 0,
            }

        entry.update({
            "output": _sanitize(text),
            "output_sha256": _sha256(text),
            "blocked": bool(blocked),
            "layer": layer,
            "reason": reason,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1) if started else None,
        })
        if extra:
            entry["extra"] = extra

        prev_hash = self.logs[-1]["entry_hash"] if self.logs else _GENESIS_HASH
        entry["prev_hash"] = prev_hash
        entry["entry_hash"] = _sha256(prev_hash + json.dumps(entry, sort_keys=True, ensure_ascii=False))
        self.logs.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Recompute the hash chain; False if any entry was altered or removed."""
        prev_hash = _GENESIS_HASH
        for entry in self.logs:
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if body.get("prev_hash") != prev_hash:
                return False
            expected = _sha256(prev_hash + json.dumps(body, sort_keys=True, ensure_ascii=False))
            if entry.get("entry_hash") != expected:
                return False
            prev_hash = entry["entry_hash"]
        return True

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
