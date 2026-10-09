"""Tests for `cswap rewake`, the StopFailure hook (rewake.py)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, rewake
from claude_swap.models import AccountSnapshot, AccountsSnapshot
from claude_swap.rewake import (
    WAKE_EXIT_CODE,
    Observation,
    RewakeDeps,
    _State,
    hook_main,
    observer_for,
    run_rewake,
    session_resumed,
)
from claude_swap.settings import AutoSwitchSettings
from claude_swap.usage_store import UsageEntry

SESSION = "0123456789abcdef"


class World:
    """A fake clock and usage store whose state changes at scheduled times."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now
        self.obs = Observation("8", False, now - 30.0)
        self.changes: list[tuple[float, object]] = []
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def at(self, offset: float, change) -> None:
        self.changes.append((self.now + offset, change))

    def observe(self) -> Observation:
        for due, change in list(self.changes):
            if self.now >= due:
                self.changes.remove((due, change))
                change()
        return self.obs

    def set(self, active, under, fetched_offset):
        """A change that makes the store report ``active`` as of now + offset."""

        def change():
            self.obs = Observation(active, under, self.now + fetched_offset)

        return change


def _payload(transcript: Path | None = None, error: str = "rate_limit") -> dict:
    payload = {
        "session_id": SESSION,
        "hook_event_name": "StopFailure",
        "error": error,
        "last_assistant_message": "You've hit your session limit · resets 8:50pm",
    }
    if transcript is not None:
        payload["transcript_path"] = str(transcript)
    return payload


def _run(world: World, state_dir: Path, payload: dict, **kwargs) -> tuple[int, str]:
    deps = RewakeDeps(
        observe=world.observe, state_dir=state_dir, clock=world.clock, sleep=world.sleep
    )
    kwargs.setdefault("poll_s", 10.0)
    kwargs.setdefault("spacing_s", 10.0)
    kwargs.setdefault("max_wait_s", 3600.0)
    return run_rewake(payload, deps, **kwargs)


class TestWaitsForUsage:
    def test_other_errors_are_not_its_business(self, tmp_path):
        world = World()
        assert _run(world, tmp_path, _payload(error="overloaded")) == (0, "")
        assert world.slept == []

    def test_wakes_once_cswap_lands_on_an_account_under_its_limit(self, tmp_path):
        world = World()
        start = world.now
        world.at(30.0, world.set("14", True, -200.0))
        code, message = _run(world, tmp_path, _payload())
        assert code == WAKE_EXIT_CODE
        assert "account 14" in message and "Continue the task" in message
        assert 30.0 <= world.now - start <= 40.0

    def test_the_account_that_stopped_needs_a_reading_after_the_stop(self, tmp_path):
        world = World()
        start = world.now

        def read_before_the_stop():
            # Under its limit, but read before the stop: that reading ran out.
            world.obs = Observation("8", True, start - 5.0)

        world.at(20.0, read_before_the_stop)
        world.at(60.0, world.set("8", True, -5.0))
        code, message = _run(world, tmp_path, _payload())
        assert code == WAKE_EXIT_CODE and "account 8" in message
        assert world.now - start >= 60.0

    def test_a_stale_reading_of_another_account_is_not_trusted(self, tmp_path):
        world = World()
        start = world.now
        world.at(10.0, world.set("14", True, -1000.0))
        world.at(90.0, world.set("14", True, -30.0))
        code, _ = _run(world, tmp_path, _payload())
        assert code == WAKE_EXIT_CODE
        assert world.now - start >= 90.0

    def test_an_account_past_its_limit_never_wakes_it(self, tmp_path, caplog):
        world = World()
        world.at(10.0, world.set("14", False, -5.0))
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            assert _run(world, tmp_path, _payload(), max_wait_s=120.0) == (0, "")
        assert any("giving up" in r.getMessage() for r in caplog.records)


class TestLeavesTheSessionAlone:
    @staticmethod
    def _transcript(tmp_path: Path) -> Path:
        path = tmp_path / "session.jsonl"
        path.write_text(
            json.dumps({"type": "assistant", "isApiErrorMessage": True, "error": "rate_limit"})
            + "\n"
        )
        return path

    @staticmethod
    def _append(path: Path, entry: dict):
        def change():
            with path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")

        return change

    def test_a_reply_since_the_stop_means_it_continued(self, tmp_path):
        transcript = self._transcript(tmp_path)
        world = World()
        world.at(20.0, self._append(transcript, {"type": "assistant", "message": {}}))
        world.at(60.0, world.set("14", True, -5.0))
        assert _run(world, tmp_path, _payload(transcript)) == (0, "")

    def test_a_typed_prompt_means_the_user_took_over(self, tmp_path):
        transcript = self._transcript(tmp_path)
        world = World()
        prompt = {"type": "user", "message": {"content": [{"type": "text", "text": "continue"}]}}
        world.at(20.0, self._append(transcript, prompt))
        world.at(60.0, world.set("14", True, -5.0))
        assert _run(world, tmp_path, _payload(transcript)) == (0, "")

    def test_tool_results_notifications_and_errors_do_not_count(self, tmp_path):
        transcript = self._transcript(tmp_path)
        world = World()
        for i, entry in enumerate([
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
            {"type": "user", "message": {"content": "<task-notification>done</task-notification>"}},
            {"type": "user", "isMeta": True, "message": {"content": "Caveat: meta"}},
            {"type": "assistant", "isApiErrorMessage": True, "error": "rate_limit"},
            {"type": "assistant", "isSidechain": True, "message": {}},
        ]):
            world.at(5.0 + i, self._append(transcript, entry))
        world.at(40.0, world.set("14", True, -5.0))
        code, _ = _run(world, tmp_path, _payload(transcript))
        assert code == WAKE_EXIT_CODE

    def test_a_newer_stop_of_the_same_session_takes_over(self, tmp_path):
        world = World()

        def newer_stop():
            _State(tmp_path).update(
                lambda d: d["sessions"][SESSION].update(token="newer")
            )

        world.at(20.0, newer_stop)
        world.at(60.0, world.set("14", True, -5.0))
        assert _run(world, tmp_path, _payload()) == (0, "")

    def test_a_session_woken_twice_recently_is_left_for_the_user(self, tmp_path):
        world = World()
        _State(tmp_path).update(
            lambda d: d["sessions"].update(
                {SESSION: {"token": "old", "stoppedAt": world.now - 200,
                           "wakes": [world.now - 600, world.now - 100]}}
            )
        )
        world.at(1.0, world.set("14", True, -5.0))
        assert _run(world, tmp_path, _payload()) == (0, "")
        assert world.slept == []

    def test_wakes_older_than_the_window_do_not_count(self, tmp_path):
        world = World()
        _State(tmp_path).update(
            lambda d: d["sessions"].update(
                {SESSION: {"token": "old", "stoppedAt": world.now - 4000,
                           "wakes": [world.now - 4000, world.now - 3700]}}
            )
        )
        world.at(1.0, world.set("14", True, -5.0))
        assert _run(world, tmp_path, _payload())[0] == WAKE_EXIT_CODE


class TestOrphanedHook:
    def test_it_stops_when_the_session_process_has_gone(self, tmp_path):
        world = World()
        alive = {"value": True}
        world.at(20.0, lambda: alive.update(value=False))
        world.at(60.0, world.set("14", True, -5.0))
        deps = RewakeDeps(
            observe=world.observe, state_dir=tmp_path, clock=world.clock,
            sleep=world.sleep, session_alive=lambda: alive["value"],
        )
        assert run_rewake(_payload(), deps, poll_s=10.0, spacing_s=10.0) == (0, "")

    def test_the_check_watches_the_shell_parent(self, tmp_path):
        """Run the real check from a child of `sh -c`, as Claude Code runs a hook."""
        import subprocess
        import sys

        script = (
            "from claude_swap.rewake import claude_process_alive_check as c; "
            "print(c()())"
        )
        out = subprocess.run(
            ["sh", "-c", f"{sys.executable} -c '{script}'; true"],
            capture_output=True, text=True, timeout=60,
        )
        assert out.stdout.strip() == "True"


class TestSpacing:
    def test_a_wake_right_after_another_waits_its_turn(self, tmp_path):
        world = World()
        start = world.now
        # cswap moves at the first poll, just as another session woke.
        world.at(1.0, world.set("14", True, -5.0))
        _State(tmp_path).update(lambda d: d.update(lastWakeAt=start + 10.0))
        code, _ = _run(world, tmp_path, _payload(), spacing_s=15.0)
        assert code == WAKE_EXIT_CODE
        assert world.now - start == pytest.approx(25.0)

    def test_two_sessions_stopped_together_wake_apart(self, tmp_path):
        # Two hook processes with their own clocks, sharing the state file.
        a, b = World(), World()
        for world in (a, b):
            world.at(1.0, world.set("14", True, -5.0))
        first = _run(a, tmp_path, dict(_payload(), session_id="aaaaaaaaaaaa"))
        second = _run(b, tmp_path, dict(_payload(), session_id="bbbbbbbbbbbb"))
        assert first[0] == second[0] == WAKE_EXIT_CODE
        assert b.now - a.now == pytest.approx(10.0)


class TestSessionResumed:
    def test_no_transcript_means_no_evidence(self, tmp_path):
        assert session_resumed(None, 0) is False
        assert session_resumed(str(tmp_path / "missing.jsonl"), 0) is False

    def test_a_half_written_line_is_skipped(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"type": "assistant", "mess')
        assert session_resumed(str(path), 0) is False

    def test_only_lines_after_the_offset_count(self, tmp_path):
        path = tmp_path / "t.jsonl"
        before = json.dumps({"type": "assistant", "message": {}}) + "\n"
        path.write_text(before)
        assert session_resumed(str(path), len(before)) is False
        assert session_resumed(str(path), 0) is True


class TestObserver:
    """The live observer judges the active account the way cswap auto does."""

    @pytest.fixture(autouse=True)
    def _limits(self, monkeypatch):
        monkeypatch.setattr(oauth, "window_thresholds", lambda: ({"5h": 92.0, "7d": 97.0}, 90.0))
        monkeypatch.setattr(oauth, "decision_windows", lambda: frozenset({"5h", "7d"}))

    @staticmethod
    def _switcher(pct5: float, tiers: dict | None = None, active: str | None = "2"):
        entry = UsageEntry(
            last_good={"five_hour": {"pct": pct5}, "seven_day": {"pct": 20.0}},
            fetched_at=1234.0,
            age_s=1.0,
        )
        account = AccountSnapshot(
            number="2", email="b@example.com", org_name="", org_uuid="",
            is_active=active == "2", kind="oauth", switchable=True, usage=entry,
        )

        class FakeSwitcher:
            def accounts_snapshot(self, fetch=None):
                assert fetch == set()  # never fetches usage itself
                return AccountsSnapshot(active_number=active, accounts=(account,), taken_at=0.0)

            def account_seat_tiers(self):
                return tiers or {}

        return FakeSwitcher()

    def test_under_its_limit(self):
        observe = observer_for(self._switcher(40.0), lambda: AutoSwitchSettings())
        assert observe() == Observation("2", True, 1234.0)

    def test_between_the_threshold_and_its_limit_is_still_under(self):
        observe = observer_for(self._switcher(91.0), lambda: AutoSwitchSettings())
        assert observe().under_limit is True

    def test_at_its_limit(self):
        observe = observer_for(self._switcher(92.0), lambda: AutoSwitchSettings())
        assert observe().under_limit is False

    def test_a_premium_seat_is_judged_against_its_scaled_limit(self):
        switcher = self._switcher(95.0, tiers={"2": "default_claude_max_5x"})
        observe = observer_for(switcher, lambda: AutoSwitchSettings())
        assert observe().under_limit is True

    def test_no_active_account(self):
        observe = observer_for(self._switcher(10.0, active=None), lambda: AutoSwitchSettings())
        assert observe() == Observation(None, False, None)


class TestHookEntry:
    def test_bad_input_wakes_nothing(self):
        assert hook_main("not json") == (0, "")
        assert hook_main("[1, 2]") == (0, "")

    def test_a_failure_wakes_nothing(self, temp_home):
        payload = json.dumps(_payload())
        with patch.object(rewake, "run_rewake", side_effect=RuntimeError("boom")):
            assert hook_main(payload) == (0, "")

    def test_the_command_prints_only_the_reminder_and_exits_2(self, capsys):
        from claude_swap import cli

        with (
            patch.object(rewake, "hook_main", return_value=(WAKE_EXIT_CODE, "go on")),
            patch("sys.stdin.read", return_value="{}"),
            pytest.raises(SystemExit) as exc,
        ):
            cli._rewake_command([])
        assert exc.value.code == WAKE_EXIT_CODE
        out = capsys.readouterr()
        assert out.err == "go on\n" and out.out == ""

    def test_the_command_is_silent_when_it_does_not_wake(self, capsys):
        from claude_swap import cli

        with (
            patch.object(rewake, "hook_main", return_value=(0, "")),
            patch("sys.stdin.read", return_value="{}"),
            pytest.raises(SystemExit) as exc,
        ):
            cli._rewake_command([])
        assert exc.value.code == 0
        out = capsys.readouterr()
        assert out.err == "" and out.out == ""

    def test_main_dispatches_rewake_before_anything_else(self):
        from claude_swap import cli

        with (
            patch.object(cli, "_rewake_command") as command,
            patch("sys.argv", ["cswap", "rewake", "--spacing", "5"]),
        ):
            cli.main()
        command.assert_called_once_with(["--spacing", "5"])
