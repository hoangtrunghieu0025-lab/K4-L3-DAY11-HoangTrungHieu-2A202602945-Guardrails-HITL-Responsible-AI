"""
Assignment 11 — Session-level risk scoring.

Per-message guardrails judge each turn in isolation, so a patient attacker can
split an extraction into turns that are each just below the bar ("which team
handles admin access?", "what format are staff passwords?", ...). This
tracker keeps one decaying score per session (user_id) and turns the
*history* into a signal:

- every guardrail event adds points (injection, off-topic, output leak,
  canary, judge verdict...); soft probing words in an allowed message add a
  little, so slow-burn reconnaissance accumulates;
- the score decays exponentially (half-life ``half_life_s``), so an honest
  customer who once tripped a filter recovers on their own;
- ``score >= elevated`` → the output layer's LLM judge is switched on for the
  session (see OutputGuardrailPlugin judge gating);
- ``score >= block`` → the pipeline pauses the session for human review
  (HITL) before calling the model.

Unlike the rate limiter's lockout (which expires after a fixed time), the
score outlives the lockout: an attacker who waits out the 5-minute pause
comes back still flagged.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from guardrails.input_guardrails import normalize_text

# Points per event. Tuned so that: one jailbreak -> elevated (judge on);
# two jailbreaks, or one jailbreak + a few probes -> blocked for review.
EVENT_WEIGHTS: dict[str, float] = {
    "injection": 35.0,
    "canary_echo": 100.0,
    "input_too_long": 15.0,
    "off_topic": 10.0,
    "soft_probe": 10.0,
    "output_leak": 50.0,
    "canary_leak": 100.0,
    "judge_unsafe": 35.0,
    "judge_error": 5.0,
    "rate_limit": 5.0,
    "abuse_lockout": 20.0,
    "egress_denied": 25.0,
}

# Events that say nothing about intent (judge was down, customer typed fast):
# they never unlock soft-probe amplification. (PII redaction is not scored at
# all: the model leaking PII is not the customer's fault.)
_NON_HOSTILE = frozenset({"soft_probe", "judge_error", "rate_limit"})

# Words that are legitimate in isolation ("I forgot my password") but, repeated
# across a session, sketch a reconnaissance pattern.
_SOFT_PROBE = re.compile(
    r"\b(password|passcode|credential|api|token|secret|admin|administrator|internal|backend|"
    r"server|database|config|configuration|instruction|prompt|rule|policy|bypass|override|"
    r"security team|employee|staff|mat khau|noi bo|quan tri)s?\b"
)


@dataclass
class _Session:
    score: float = 0.0
    updated_at: float = 0.0
    events: list = field(default_factory=list)
    flagged: bool = False


class SessionRiskTracker:
    """Decaying per-session risk score. Observer only; the pipeline enforces."""

    name = "session_risk"

    def __init__(self, *, half_life_s: float = 600.0, elevated_threshold: float = 25.0,
                 block_threshold: float = 70.0, max_events: int = 50, clock=time.time):
        if not 0 < elevated_threshold < block_threshold:
            raise ValueError("need 0 < elevated_threshold < block_threshold")
        self.half_life_s = half_life_s
        self.elevated_threshold = elevated_threshold
        self.block_threshold = block_threshold
        self.max_events = max_events
        self._clock = clock
        self._sessions: dict[str, _Session] = {}

    def _session(self, user_id: str) -> _Session:
        s = self._sessions.setdefault(user_id, _Session(updated_at=self._clock()))
        now = self._clock()
        if s.score and now > s.updated_at:
            s.score *= math.pow(0.5, (now - s.updated_at) / self.half_life_s)
            if s.score < 0.5:
                s.score = 0.0
        s.updated_at = now
        return s

    def score(self, user_id: str) -> float:
        return round(self._session(user_id).score, 2)

    def level(self, user_id: str) -> str:
        sc = self.score(user_id)
        if sc >= self.block_threshold:
            return "blocked"
        if sc >= self.elevated_threshold:
            return "elevated"
        return "normal"

    def should_block(self, user_id: str) -> bool:
        return self.score(user_id) >= self.block_threshold

    def record(self, user_id: str, event: str, *, weight: float | None = None) -> float:
        """Add ``event`` to the session; returns the new score."""
        s = self._session(user_id)
        points = EVENT_WEIGHTS.get(event, 0.0) if weight is None else weight
        if points <= 0:
            return round(s.score, 2)
        s.score += points
        s.events.append({"t": round(s.updated_at, 3), "event": event, "points": points,
                         "score": round(s.score, 2)})
        del s.events[:-self.max_events]
        return round(s.score, 2)

    def observe_message(self, user_id: str, text: str) -> float:
        """Soft signal for messages the input guardrail let through.

        Soft probes only *amplify* a session that already showed hostile
        intent (a hard event). On their own they are capped below the
        elevated threshold: a customer asking five questions about resetting
        a password must never be escalated, judged or held for review.
        """
        if not _SOFT_PROBE.search(normalize_text(text)):
            return self.score(user_id)
        s = self._session(user_id)
        if any(e["event"] not in _NON_HOSTILE for e in s.events):
            return self.record(user_id, "soft_probe")
        headroom = (self.elevated_threshold - 1) - s.score
        return self.record(user_id, "soft_probe", weight=min(EVENT_WEIGHTS["soft_probe"], headroom))

    def mark_flagged(self, user_id: str) -> bool:
        """Mark session as sent to human review; True the first time only."""
        s = self._session(user_id)
        first = not s.flagged
        s.flagged = True
        return first

    def snapshot(self, user_id: str) -> dict:
        s = self._session(user_id)
        return {
            "user_id": user_id,
            "score": round(s.score, 2),
            "level": self.level(user_id),
            "flagged_for_review": s.flagged,
            "events": [e["event"] for e in s.events],
        }

    def reset(self, user_id: str) -> None:
        """Clear a session (e.g. after a human reviewer cleared it)."""
        self._sessions.pop(user_id, None)

    def high_risk_sessions(self) -> list[str]:
        return [uid for uid in list(self._sessions) if self.score(uid) >= self.elevated_threshold]
