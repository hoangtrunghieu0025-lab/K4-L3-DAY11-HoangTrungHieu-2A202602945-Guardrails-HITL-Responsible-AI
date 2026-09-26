"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).

On top of the plain sliding window it keeps a second, much smaller window of
*guardrail violations* per user: an attacker probing with jailbreaks gets
locked out for ``lockout_seconds`` after ``max_violations`` blocked attempts,
so they cannot keep iterating on prompts at full speed.
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
        *,
        max_violations: int = 3,
        lockout_seconds: int = 300,
        clock=time.time,
    ):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_violations = max_violations
        self.lockout_seconds = lockout_seconds
        self._clock = clock
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.violations: dict[str, deque] = defaultdict(deque)
        self.locked_until: dict[str, float] = {}
        self.blocked_count = 0
        self.lockout_count = 0
        self.total_count = 0
        self.last_block_reason: str | None = None

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    def record_violation(self, user_id: str) -> bool:
        """Register a guardrail block for ``user_id``.

        Returns True if this violation triggered a lockout.
        """
        now = self._clock()
        strikes = self.violations[user_id]
        while strikes and strikes[0] <= now - self.window_seconds:
            strikes.popleft()
        strikes.append(now)
        if len(strikes) >= self.max_violations and self.locked_until.get(user_id, 0) <= now:
            self.locked_until[user_id] = now + self.lockout_seconds
            self.lockout_count += 1
            return True
        return False

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        self.last_block_reason = None
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = self._clock()

        locked_until = self.locked_until.get(user_id, 0)
        if locked_until > now:
            self.blocked_count += 1
            self.last_block_reason = "abuse_lockout"
            return self._block_response(
                "Too many blocked requests from this session. "
                f"Access is paused for {locked_until - now:.0f}s."
            )

        window = self.user_windows[user_id]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self.last_block_reason = "rate_limit"
            return self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s."
            )

        # Only admitted requests consume a slot, so a flood of rejected
        # requests cannot extend the caller's own penalty indefinitely.
        window.append(now)
        return None
