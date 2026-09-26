"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str
    severity: str = "warning"


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    # A single secret reaching the output layer means the model tried to leak:
    # always page someone, even though the output guard caught it.
    output_leak_threshold: int = 1
    lockout_threshold: int = 1
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    injection_blocks: int = 0
    topic_blocks: int = 0
    output_leak_blocks: int = 0
    output_redactions: int = 0
    abuse_lockouts: int = 0
    llm_errors: int = 0
    egress_checks: int = 0
    egress_denied: int = 0
    blocks_by_layer: dict = field(default_factory=dict)

    def record_request(
        self,
        *,
        blocked: bool,
        layer: str | None,
        reason: str | None = None,
        redacted: bool = False,
    ) -> None:
        """Update counters for one request that went through the pipeline."""
        self.total_requests += 1
        if redacted:
            self.output_redactions += 1
        if reason == "llm_error":
            self.llm_errors += 1
        if not blocked:
            return
        self.blocked_requests += 1
        key = layer or "unknown"
        self.blocks_by_layer[key] = self.blocks_by_layer.get(key, 0) + 1
        if reason in ("rate_limit", "abuse_lockout"):
            self.rate_limit_hits += 1
        if reason == "abuse_lockout":
            self.abuse_lockouts += 1
        elif reason == "injection":
            self.injection_blocks += 1
        elif reason == "off_topic":
            self.topic_blocks += 1
        elif layer == "output_guardrail":
            self.output_leak_blocks += 1

    def record_judge(self, *, safe: bool) -> None:
        self.judge_checks += 1
        if not safe:
            self.judge_fails += 1

    def record_egress(self, *, allowed: bool) -> None:
        self.egress_checks += 1
        if not allowed:
            self.egress_denied += 1

    def _raise(self, metric: str, value: float, threshold: float, message: str, severity: str):
        # One alert per metric: re-running check_metrics must not spam duplicates.
        for alert in self.alerts:
            if alert.metric == metric:
                alert.value, alert.message, alert.severity = value, message, severity
                return
        self.alerts.append(Alert(metric, value, threshold, message, severity))

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        snap = self.snapshot()

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            self._raise(
                "block_rate", round(snap["block_rate"], 3), self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} > {self.block_rate_threshold:.0%}: "
                "possible attack campaign or over-blocking of real customers.",
                "warning",
            )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self._raise(
                "rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
                f"{self.rate_limit_hits} rate-limit hits: flooding / cost-abuse attempt.",
                "warning",
            )
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            self._raise(
                "judge_fail_rate", round(snap["judge_fail_rate"], 3), self.judge_fail_rate_threshold,
                f"LLM judge failing {snap['judge_fail_rate']:.0%} of responses.",
                "warning",
            )
        if self.output_leak_blocks >= self.output_leak_threshold:
            self._raise(
                "output_leak_blocks", self.output_leak_blocks, self.output_leak_threshold,
                f"{self.output_leak_blocks} response(s) contained protected data and were "
                "suppressed by the output guardrail — an input bypass reached the model.",
                "critical",
            )
        if self.abuse_lockouts >= self.lockout_threshold:
            self._raise(
                "abuse_lockouts", self.abuse_lockouts, self.lockout_threshold,
                f"{self.abuse_lockouts} session(s) locked out after repeated guardrail violations.",
                "warning",
            )
        if self.egress_denied:
            self._raise(
                "egress_denied", self.egress_denied, 1,
                f"{self.egress_denied} outbound request(s) denied by the egress allowlist.",
                "critical",
            )
        return self.alerts

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **self.snapshot()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "injection_blocks": self.injection_blocks,
            "topic_blocks": self.topic_blocks,
            "output_leak_blocks": self.output_leak_blocks,
            "output_redactions": self.output_redactions,
            "abuse_lockouts": self.abuse_lockouts,
            "llm_errors": self.llm_errors,
            "egress_checks": self.egress_checks,
            "egress_denied": self.egress_denied,
            "blocks_by_layer": dict(self.blocks_by_layer),
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
