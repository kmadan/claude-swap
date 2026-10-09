"""Wake a Claude Code session that a usage limit stopped, once cswap can serve it.

Claude Code's own automatic continue waits for the reset of the window that
stopped the session, and a session hosted by an editor extension may not start
that wait at all. When cswap has already moved the login to an account with
room, both leave the session idle for hours that the fleet could have covered.

``cswap rewake`` runs as a Claude Code ``StopFailure`` hook with
``asyncRewake: true``. Claude Code starts it in the background when a turn ends
on an API error and passes the hook input as JSON on stdin. When the error is a
rate limit, the command waits until the account cswap has active is under its
switch limit, then exits with ``WAKE_EXIT_CODE``. The registered command maps
that status to the 2 that wakes the idle session, and the text written to
stderr reaches Claude as a reminder to continue. Every other outcome exits 0,
which wakes nothing.

The wake status is not 2 itself because argparse exits 2 on a usage error, and
so does any cswap without this command. Mapped through the registration, such
a failure leaves the session alone instead of waking it, and a wake that
failed the same way on every stop cannot turn into a loop.

The account active when the stop happened only counts once a reading taken
after the stop shows it under its limit, because the reading before the stop is
the one that ran out. Another account counts on any reading recent enough to
trust. The command does not fetch usage itself: ``cswap auto`` keeps the store
current, and this reads it.

It stays quiet when the session needs no help. A session that has produced a
new reply, or received a typed prompt, since the stop is left alone, and when
the same session stops again the newer hook process takes over from the older
one. Wakes are spaced so sessions stopped together do not all land on the new
account at once, and a session woken twice within half an hour without
recovering is left for the user, so a wake that keeps hitting a limit cannot
loop.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

_logger = logging.getLogger("claude-swap")

# Mapped to Claude Code's wake status 2 by the registered command; see above.
WAKE_EXIT_CODE = 42
STATE_FILE_NAME = "rewake_state.json"
LOCK_FILE_NAME = "rewake.lock"

DEFAULT_ERRORS = ("rate_limit",)
DEFAULT_POLL_S = 10.0
DEFAULT_SPACING_S = 10.0
# Just under the six-hour hook timeout the README registers, so the command
# exits cleanly rather than being killed.
DEFAULT_MAX_WAIT_S = 5.75 * 3600

# A reading of an account other than the one that stopped is trusted this long.
# Candidates are read less often than the active account, and an account that
# has not been used since its reading cannot have filled up in the meantime.
OTHER_ACCOUNT_MAX_AGE_S = 900.0
# Wakes allowed per session inside the window before the session is left alone.
MAX_WAKES = 2
WAKE_WINDOW_S = 1800.0
# Session records older than this are dropped from the state file.
STATE_RETENTION_S = 86400.0


@dataclass(frozen=True)
class Observation:
    """What the usage store says about the account cswap has active."""

    active: str | None
    under_limit: bool
    fetched_at: float | None


@dataclass
class RewakeDeps:
    """Everything the wait reads from the outside world, injectable for tests."""

    observe: Callable[[], Observation]
    state_dir: Path
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    # Whether the Claude Code process that ran the hook is still there.
    session_alive: Callable[[], bool] = lambda: True


_SHELLS = {"sh", "bash", "dash", "zsh", "ash", "ksh"}


def claude_process_alive_check() -> Callable[[], bool]:
    """A check that the Claude Code process which started this hook still runs.

    Claude Code runs a shell-form hook as ``sh -c <command>``, so the process
    to watch is the shell's parent; an exec-form hook's parent is Claude Code
    itself. When the session's process exits (its editor panel closed), the
    hook is orphaned and the shell stays behind it, so watching the direct
    parent would never notice. Reads ``/proc``; where that is unavailable the
    check always passes, which only means the wait runs to its own limit.
    """

    def stat_ppid(pid: int) -> int | None:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                # The command name is parenthesised and may contain spaces.
                return int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return None

    def comm(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/comm") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    parent = os.getppid()
    target = parent
    if comm(parent) in _SHELLS:
        target = stat_ppid(parent) or parent
    if target <= 1 or not os.path.exists(f"/proc/{target}"):
        return lambda: True
    return lambda: os.path.exists(f"/proc/{target}")


def session_resumed(transcript_path: str | None, offset: int | None) -> bool:
    """Whether the session moved on after the stop without needing a wake.

    Reads the transcript from ``offset``, the size it had when the hook
    started. A reply that is not an API error means a turn has succeeded since,
    and a typed prompt means the user has taken over; either way a wake would
    land in a session that is already working. Subagent lines, tool results,
    task notifications and meta entries are not evidence of either.
    """
    if not transcript_path or offset is None:
        return False
    try:
        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return False
    for raw in data.splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue  # a line still being written
        if not isinstance(entry, dict) or entry.get("isSidechain"):
            continue
        kind = entry.get("type")
        if kind == "assistant" and not entry.get("isApiErrorMessage"):
            return True
        if kind == "user" and _is_typed_prompt(entry):
            return True
    return False


def _is_typed_prompt(entry: dict) -> bool:
    if entry.get("isMeta"):
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        text = content.strip()
        return bool(text) and not text.startswith("<")
    if isinstance(content, list):
        kinds = {part.get("type") for part in content if isinstance(part, dict)}
        if "tool_result" in kinds:
            return False
        texts = [
            part.get("text", "").strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return any(text and not text.startswith("<") for text in texts)
    return False


class _State:
    """The shared state file, read and written under one lock per change."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / STATE_FILE_NAME
        self.lock_path = state_dir / LOCK_FILE_NAME

    def update(self, change: Callable[[dict], object]) -> object:
        return self._locked(change, write=True)

    def read(self, query: Callable[[dict], object]) -> object:
        return self._locked(query, write=False)

    def _locked(self, fn: Callable[[dict], object], *, write: bool) -> object:
        lock = FileLock(self.lock_path, timeout=30.0)
        if not lock.acquire():
            raise TimeoutError(f"could not lock {self.lock_path}")
        try:
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            if not isinstance(data.get("sessions"), dict):
                data["sessions"] = {}
            result = fn(data)
            if write:
                atomic_write_json(self.path, data)
            return result
        finally:
            lock.release()


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _recent(wakes: object, now: float) -> list[float]:
    if not isinstance(wakes, list):
        return []
    return [w for w in wakes if isinstance(w, (int, float)) and now - w < WAKE_WINDOW_S]


def run_rewake(
    payload: dict,
    deps: RewakeDeps,
    *,
    errors: Iterable[str] = DEFAULT_ERRORS,
    poll_s: float = DEFAULT_POLL_S,
    spacing_s: float = DEFAULT_SPACING_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> tuple[int, str]:
    """Wait for usage on behalf of one stopped session.

    Returns ``(exit_code, message)``: ``WAKE_EXIT_CODE`` with the reminder for
    Claude when the session should continue, otherwise ``0`` and no message.
    """
    if payload.get("hook_event_name") != "StopFailure":
        return 0, ""
    if payload.get("error") not in set(errors):
        return 0, ""
    if payload.get("agent_id"):
        # A subagent's stop. The main thread receives the failure as the
        # subagent's result, and when its own next request is refused it stops
        # with a StopFailure of its own, which is the one to wake. Waking the
        # session for the subagent could start a turn nobody asked for.
        return 0, ""
    session = payload.get("session_id")
    if not isinstance(session, str) or not session:
        return 0, ""
    short = session[:8]
    transcript = payload.get("transcript_path")
    offset: int | None = None
    if isinstance(transcript, str) and transcript:
        try:
            offset = Path(transcript).stat().st_size
        except OSError:
            offset = None

    state = _State(deps.state_dir)
    started = deps.clock()
    stopped_on = deps.observe().active
    token = secrets.token_hex(8)

    def register(data: dict) -> bool:
        sessions = data["sessions"]
        for sid in [
            s
            for s, rec in sessions.items()
            if not isinstance(rec, dict)
            or started - (_number(rec.get("stoppedAt")) or 0.0) > STATE_RETENTION_S
        ]:
            del sessions[sid]
        record = sessions.get(session) if isinstance(sessions.get(session), dict) else {}
        wakes = _recent(record.get("wakes"), started)
        sessions[session] = {
            "token": token,
            "stoppedAt": started,
            "account": stopped_on,
            "wakes": wakes,
        }
        return len(wakes) >= MAX_WAKES

    if state.update(register):
        _logger.warning(
            "Rewake: session %s stopped again after %d wakes in %.0f minutes; "
            "leaving it for the user",
            short, MAX_WAKES, WAKE_WINDOW_S / 60,
        )
        return 0, ""
    _logger.info(
        "Rewake: session %s stopped (%s) on account %s; waiting for an account "
        "under its limit",
        short, payload.get("error"), stopped_on or "unknown",
    )

    def owner(data: dict) -> bool:
        record = data["sessions"].get(session)
        return isinstance(record, dict) and record.get("token") == token

    def claim(now: float) -> Callable[[dict], float | None]:
        def change(data: dict) -> float | None:
            if not owner(data):
                return None
            last = _number(data.get("lastWakeAt")) or 0.0
            slot = max(now, last + spacing_s)
            data["lastWakeAt"] = slot
            return slot

        return change

    def record_wake(at: float) -> Callable[[dict], None]:
        def change(data: dict) -> None:
            record = data["sessions"].get(session)
            if isinstance(record, dict):
                record["wakes"] = _recent(record.get("wakes"), at) + [at]

        return change

    def ready(obs: Observation, now: float) -> bool:
        if obs.active is None or not obs.under_limit or obs.fetched_at is None:
            return False
        if obs.active == stopped_on:
            # The reading before the stop is the one that ran out.
            return obs.fetched_at > started
        return now - obs.fetched_at <= OTHER_ACCOUNT_MAX_AGE_S

    def leave_alone() -> str | None:
        if not deps.session_alive():
            return "the Claude Code process that ran it has exited"
        if not state.read(owner):
            return "a newer stop of the same session took over"
        if session_resumed(transcript, offset):
            return "the session continued without a wake"
        return None

    while True:
        now = deps.clock()
        if now - started > max_wait_s:
            _logger.info(
                "Rewake: session %s waited %.0f minutes without an account under "
                "its limit; giving up",
                short, (now - started) / 60,
            )
            return 0, ""
        reason = leave_alone()
        if reason:
            _logger.info("Rewake: session %s left alone: %s", short, reason)
            return 0, ""
        obs = deps.observe()
        if ready(obs, now):
            slot = state.update(claim(now))
            if slot is None:
                _logger.info(
                    "Rewake: session %s left alone: a newer stop of the same "
                    "session took over",
                    short,
                )
                return 0, ""
            if slot > now:
                deps.sleep(slot - now)
            reason = leave_alone()
            if reason:
                _logger.info("Rewake: session %s left alone: %s", short, reason)
                return 0, ""
            obs = deps.observe()
            if ready(obs, deps.clock()):
                woke = deps.clock()
                state.update(record_wake(woke))
                _logger.info(
                    "Rewake: waking session %s on account %s, %.0fs after the stop",
                    short, obs.active, woke - started,
                )
                return WAKE_EXIT_CODE, (
                    f"Usage is available again: cswap is now on account "
                    f"{obs.active}, which is under its limit. Continue the task "
                    f"you were working on."
                )
        deps.sleep(poll_s)


def observer_for(switcher, load_settings: Callable[[], object]) -> Callable[[], Observation]:
    """An ``observe`` callable reading the live usage store through ``switcher``.

    Settings are reloaded on every call, so a change made with ``cswap config
    set`` during a long wait applies to it, as it does to ``cswap auto``.
    """
    from claude_swap import oauth
    from claude_swap.seats import effective_weights
    from claude_swap.settings import parse_account_weights, parse_model_names

    def observe() -> Observation:
        settings = load_settings()
        snapshot = switcher.accounts_snapshot(fetch=set())
        active = snapshot.active_number
        if active is None:
            return Observation(None, False, None)
        account = next((a for a in snapshot.accounts if a.number == active), None)
        entry = account.usage if account is not None else None
        usage = entry.last_good if entry is not None else None
        if not isinstance(usage, dict):
            return Observation(active, False, None)
        try:
            tiers = switcher.account_seat_tiers()
        except Exception:  # a profile read must not end the wait
            tiers = {}
        weights = effective_weights(parse_account_weights(settings.account_weights), tiers)
        headroom = oauth.account_headroom(
            usage,
            parse_model_names(settings.model),
            weights.get(active, 1.0),
            settings.threshold,
        )
        under = (
            headroom is not None
            and headroom > 0
            and (100.0 - headroom) < settings.threshold
        )
        return Observation(active, under, entry.fetched_at)

    return observe


def hook_main(
    stdin_text: str,
    *,
    errors: Iterable[str] = DEFAULT_ERRORS,
    poll_s: float = DEFAULT_POLL_S,
    spacing_s: float = DEFAULT_SPACING_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> tuple[int, str]:
    """Run the hook against the real usage store. Any failure wakes nothing."""
    try:
        payload = json.loads(stdin_text or "{}")
        if not isinstance(payload, dict):
            return 0, ""
        from claude_swap import paths
        from claude_swap.settings import load_settings
        from claude_swap.switcher import ClaudeAccountSwitcher

        root = paths.get_backup_root()
        switcher = ClaudeAccountSwitcher()
        deps = RewakeDeps(
            observe=observer_for(switcher, lambda: load_settings(root)),
            state_dir=root,
            session_alive=claude_process_alive_check(),
        )
        return run_rewake(
            payload,
            deps,
            errors=errors,
            poll_s=poll_s,
            spacing_s=spacing_s,
            max_wait_s=max_wait_s,
        )
    except Exception:
        _logger.exception("Rewake: failed; the session is left alone")
        return 0, ""
