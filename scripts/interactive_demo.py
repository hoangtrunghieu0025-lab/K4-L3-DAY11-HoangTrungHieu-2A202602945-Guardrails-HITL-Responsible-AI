#!/usr/bin/env python3
"""
Interactive terminal demo — type a prompt, watch it go through the Blue
defense pipeline and see whether anything leaked.

    python scripts/interactive_demo.py                 # live Blue model (OpenRouter)
    python scripts/interactive_demo.py --offline       # no API call: canned model reply
    python scripts/interactive_demo.py --show-raw      # also print the raw model output

Each turn shows: the layer that decided, the reason, the session risk score,
the LLM-judge verdict (if the session was risky enough to consult it), a leak
scan of the *raw* model output (before the output guardrail) and the final
reply the customer would see.

Commands (type at the prompt):
    /model <text>   pretend the model answered <text> (tests the output layer)
    /user <id>      switch to another session / user id
    /session        show this session's risk score and events
    /reset          clear this session (as if a human reviewer approved it)
    /stats          monitoring counters + alerts
    /canary         show the canary fingerprint (and how to trip it)
    /help           this help
    /quit           exit

Nothing is written to outputs/ unless you pass --export DIR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
try:
    sys.stdin.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from assignment.pipeline import (  # noqa: E402
    DefensePipeline,
    build_observability,
    build_production_plugins,
)
from core.config import blue_provider_label, get_blue_model_endpoint, get_openrouter_api_key  # noqa: E402
from guardrails.canary import detect_canary  # noqa: E402
from guardrails.output_guardrails import detect_secret_leak  # noqa: E402

OFFLINE_REPLY = (
    "Thank you for contacting VinBank. The 12-month savings rate is 4.25% per year. "
    "You can manage transfers and cards in the VinBank mobile app."
)


class Style:
    def __init__(self, enabled: bool):
        if enabled and os.name == "nt":
            os.system("")  # enable ANSI escape processing on Windows consoles
        self.on = enabled

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def red(self, t): return self._c("1;31", t)
    def green(self, t): return self._c("1;32", t)
    def yellow(self, t): return self._c("1;33", t)
    def cyan(self, t): return self._c("36", t)
    def dim(self, t): return self._c("2", t)
    def bold(self, t): return self._c("1", t)


class Demo:
    def __init__(self, *, offline: bool, use_judge: bool, show_raw: bool, user: str, style: Style):
        plugins = build_production_plugins(use_llm_judge=use_judge)
        audit, monitor = build_observability()
        self.pipe = DefensePipeline(plugins, audit, monitor, llm_retries=2)
        self.offline = offline
        self.show_raw = show_raw
        self.user = user
        self.s = style
        self._forced_reply: str | None = None
        self.last_raw: str | None = None

        real_call = self.pipe._call_llm

        async def traced_call(text: str) -> str:
            if self._forced_reply is not None:
                raw = self._forced_reply
            elif self.offline:
                raw = OFFLINE_REPLY
            else:
                raw = await real_call(text)
            self.last_raw = raw
            return raw

        self.pipe._call_llm = traced_call

    # ------------------------------------------------------------ output

    def banner(self):
        s = self.s
        out = self.pipe.output_guardrail
        mode = "OFFLINE (canned reply)" if self.offline else f"LIVE {blue_provider_label()} → {get_blue_model_endpoint()}"
        print(s.bold("=" * 72))
        print(s.bold(" VinBank Blue — interactive defense demo"))
        print(s.bold("=" * 72))
        print(f" Model      : {mode}")
        print(f" Layers     : {' → '.join(p.name for p in self.pipe.plugins)} (+ session_risk gate)")
        judge = f"on, {out.judge_mode}, risk ≥ {out.judge_risk_threshold:g}" if out and out.use_llm_judge else "off"
        print(f" LLM judge  : {judge}")
        print(f" Canary     : planted (fingerprint {self.pipe.canary_fingerprint})")
        print(f" Session    : {self.user}")
        print(s.dim(" Type a prompt, or /help for commands. Ctrl+C or /quit to exit."))
        print()

    def show_result(self, row: dict, elapsed: float, simulated: bool):
        s = self.s
        if row["blocked"]:
            status = s.red("BLOCKED")
        elif row.get("redacted"):
            status = s.yellow("REDACTED")
        else:
            status = s.green("PASSED")
        level = row.get("session_level", "normal")
        level_txt = {"normal": s.green, "elevated": s.yellow, "blocked": s.red}.get(level, str)(level)

        print(f"  {s.bold('Decision')}   : {status}  layer={row.get('layer') or '-'}  reason={row.get('reason') or '-'}")
        if row.get("issues"):
            print(f"  {s.bold('Signals')}    : {', '.join(map(str, row['issues']))}")
        print(f"  {s.bold('Session')}    : risk={row.get('session_risk', 0):.1f}  level={level_txt}")
        if row.get("judge_verdict"):
            v = row["judge_verdict"]
            print(f"  {s.bold('LLM judge')}  : {(s.green if v == 'SAFE' else s.red)(v)}")

        # Leak scan of what the model produced *before* the output guardrail.
        if self.last_raw is None:
            print(f"  {s.bold('Model')}      : {s.dim('not called (stopped before the LLM)')}")
        else:
            signals = detect_secret_leak(self.last_raw)
            tag = " (simulated via /model)" if simulated else ""
            if signals:
                kind = "CANARY — system prompt exfiltrated" if detect_canary(self.last_raw) else "protected data"
                print(f"  {s.bold('Leak scan')}  : {s.red('LEAK ATTEMPT in raw model output' + tag)} → {kind}")
                print(f"               signals={', '.join(signals)}")
                verdict = s.green("contained — customer did NOT receive it") if row["blocked"] else s.red("NOT CONTAINED")
                print(f"               {verdict}")
            else:
                print(f"  {s.bold('Leak scan')}  : {s.green('no protected data in raw model output' + tag)}")
            if self.show_raw:
                print(f"  {s.bold('Raw model')}  : {s.dim(self.last_raw)}")

        print(f"  {s.bold('Reply')}      : {s.cyan(row.get('response_preview', ''))}")
        print(s.dim(f"  ({elapsed * 1000:.0f} ms)"))
        print()

    # ------------------------------------------------------------ commands

    def cmd_help(self, _):
        print(__doc__.split("Commands (type at the prompt):", 1)[1].split("Nothing is written", 1)[0])

    def cmd_user(self, arg):
        if not arg:
            print(f"  current session: {self.user}\n")
            return
        self.user = arg.strip()
        print(f"  switched to session {self.s.bold(self.user)}\n")

    def cmd_reset(self, _):
        """Simulate a human reviewer clearing this session."""
        self.pipe.session_risk.reset(self.user)
        rl = self.pipe.rate_limiter
        if rl:
            for table in (rl.user_windows, rl.violations, rl.locked_until):
                table.pop(self.user, None)
        print(f"  session {self.s.bold(self.user)} cleared (risk score, rate limit, lockout)\n")

    def cmd_session(self, _):
        print("  " + json.dumps(self.pipe.session_risk.snapshot(self.user), ensure_ascii=False) + "\n")

    def cmd_stats(self, _):
        self.pipe.monitor.check_metrics()
        snap = self.pipe.monitor.snapshot()
        keys = ["total_requests", "blocked_requests", "injection_blocks", "topic_blocks",
                "output_leak_blocks", "output_redactions", "canary_triggers", "judge_checks",
                "judge_fails", "judge_errors", "session_risk_blocks", "rate_limit_hits", "abuse_lockouts"]
        for k in keys:
            print(f"  {k:<20} {snap[k]}")
        for a in snap["alerts"]:
            color = self.s.red if a["severity"] == "critical" else self.s.yellow
            print(f"  {color('ALERT ' + a['severity'].upper())} {a['metric']}: {a['message']}")
        print()

    def cmd_canary(self, _):
        print(f"  Canary fingerprint: {self.pipe.canary_fingerprint} (token itself is never printed).")
        print("  To see it trip without a real leak:  /model __CANARY__")
        print("  (__CANARY__ is replaced by the live token only inside the simulated model reply.)\n")

    async def turn(self, text: str):
        simulated = False
        self._forced_reply = None
        self.last_raw = None
        if text.startswith("/model"):
            reply = text[len("/model"):].strip()
            if not reply:
                print("  usage: /model <text the model 'said'>\n")
                return
            self._forced_reply = reply.replace("__CANARY__", self.pipe.canary)
            simulated = True
            # the customer message for this turn is a benign placeholder
            text = "What is the savings interest rate?"
        started = time.perf_counter()
        row = await self.pipe.process(text, user_id=self.user)
        self.show_result(row, time.perf_counter() - started, simulated)
        self._forced_reply = None

    async def loop(self):
        commands = {
            "/help": self.cmd_help, "/user": self.cmd_user, "/session": self.cmd_session,
            "/stats": self.cmd_stats, "/canary": self.cmd_canary, "/reset": self.cmd_reset,
        }
        self.banner()
        while True:
            try:
                text = input(self.s.bold(f"[{self.user}] > "))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            text = text.strip()
            if not text:
                continue
            if text in ("/quit", "/exit", "/q"):
                break
            name, _, arg = text.partition(" ")
            if name in commands:
                commands[name](arg)
                continue
            try:
                await self.turn(text)
            except Exception as exc:  # keep the demo alive
                print(self.s.red(f"  error: {type(exc).__name__}: {exc}\n"))


def main():
    parser = argparse.ArgumentParser(description="Interactive Blue defense pipeline demo")
    parser.add_argument("--offline", action="store_true", help="do not call the model; use a canned reply")
    parser.add_argument("--no-judge", action="store_true", help="disable the risk-gated LLM judge")
    parser.add_argument("--show-raw", action="store_true", help="print raw model output (may contain secrets!)")
    parser.add_argument("--user", default="demo-user", help="initial session / user id")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--export", type=Path, default=None,
                        help="on exit, write audit_log.json + metrics.json into this directory")
    args = parser.parse_args()

    offline = args.offline
    if not offline and not get_openrouter_api_key():
        print("OPENROUTER_API_KEY not set — falling back to --offline mode.")
        offline = True

    demo = Demo(offline=offline, use_judge=not args.no_judge and not offline, show_raw=args.show_raw,
                user=args.user, style=Style(not args.no_color and sys.stdout.isatty()))
    asyncio.run(demo.loop())

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        demo.pipe.audit.export_json(str(args.export / "audit_log.json"))
        demo.pipe.monitor.export_json(str(args.export / "metrics.json"))
        print(f"Exported audit_log.json + metrics.json to {args.export}")
    print("bye.")


if __name__ == "__main__":
    main()
