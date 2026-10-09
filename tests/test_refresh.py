"""``cswap refresh``: an on-request fetch of one account, paced by its budget."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from claude_swap import cli, oauth
from claude_swap.credentials import ActiveCredentials
from claude_swap.exceptions import ValidationError
from claude_swap.switcher import ClaudeAccountSwitcher, refresh_refusal
from claude_swap.usage_store import (
    AUTH_DEAD_STRIKES,
    FetchRecord,
    UsageEntry,
    _row_eligible,
)

NOW = 1_000_000.0
OLD = {"five_hour": {"pct": 25.0}, "seven_day": {"pct": 10.0}}
NEW = {"five_hour": {"pct": 40.0}, "seven_day": {"pct": 12.0}}


class TestForcedReservation:
    """``force`` skips freshness and the plan, and nothing else."""

    @staticmethod
    def _row(**fields):
        row = {"fetchedAt": NOW - 30, "nextPollAt": NOW + 300}  # fresh and not due
        row.update(fields)
        return row

    def test_force_skips_freshness_and_the_plan(self):
        assert not _row_eligible(self._row(), NOW, respect_plans=False)
        assert _row_eligible(self._row(), NOW, respect_plans=False, force=True)

    def test_force_still_honours_backoff(self):
        row = self._row(backoffUntil=NOW + 60)
        assert not _row_eligible(row, NOW, respect_plans=False, force=True)

    def test_force_still_honours_an_in_flight_fetch(self):
        row = self._row(claimUntil=NOW + 30)
        assert not _row_eligible(row, NOW, respect_plans=False, force=True)

    def test_force_still_honours_a_dead_login(self):
        row = self._row(authDeadStrikes=AUTH_DEAD_STRIKES)
        assert not _row_eligible(row, NOW, respect_plans=False, force=True)


class TestRefreshRefusal:
    def test_a_read_over_a_minute_old_may_refresh(self):
        assert refresh_refusal(UsageEntry(fetched_at=NOW - 61), NOW) is None

    def test_a_read_within_the_minute_waits(self):
        reason = refresh_refusal(UsageEntry(fetched_at=NOW - 30), NOW)
        assert reason == "it was read 30s ago; try again in 30s"

    def test_a_running_backoff_declines(self):
        entry = UsageEntry(fetched_at=NOW - 600, backoff_until=NOW + 120)
        assert refresh_refusal(entry, NOW).startswith("it is backing off")

    def test_a_429_within_the_hour_declines(self):
        entry = UsageEntry(fetched_at=NOW - 600, last_429_at=NOW - 1800)
        reason = refresh_refusal(entry, NOW)
        assert "rate-limited it 30 min ago" in reason
        assert reason.endswith("try again in 30 min")

    def test_a_429_over_an_hour_old_does_not(self):
        entry = UsageEntry(fetched_at=NOW - 600, last_429_at=NOW - 3601)
        assert refresh_refusal(entry, NOW) is None


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def refreshable(temp_home, mock_claude_config, sample_sequence_data):
    """A switcher whose account 2 was read 120 s ago, inside the serve window.

    A plan in the future as well, so no on-demand pass would fetch it.
    """
    switcher = ClaudeAccountSwitcher()
    switcher._setup_directories()
    switcher._write_json(switcher.sequence_file, sample_sequence_data)
    clock = _Clock()
    switcher._usage_store.clock = clock
    identity = {"2": ("account2@example.com", "")}
    switcher._usage_store.record({"2": FetchRecord(usage=OLD)}, identity)
    switcher._usage_store.set_poll_plan({"2": (NOW + 600, 300.0)}, identity)
    clock.now += 120
    creds = json.dumps({"claudeAiOauth": {"accessToken": "sk-backup"}})
    with patch.object(
        switcher, "_read_active_credentials",
        return_value=ActiveCredentials(creds, False),
    ), patch.object(switcher, "_read_account_credentials", return_value=creds):
        yield switcher, clock, identity


def _fetch_returning(usage):
    return patch(
        "claude_swap.oauth.try_fetch_usage_for_account",
        return_value=oauth.UsageOutcome(usage),
    )


class TestRefreshCommand:
    def test_an_on_demand_pass_would_not_fetch_here(self, refreshable):
        """The control: inside the serve window, `cswap list` serves the store."""
        switcher, _clock, identity = refreshable
        before = switcher._usage_store.entries(identity)["2"].fetched_at
        with _fetch_returning(NEW):
            switcher._collect_usage_entries(switcher._build_accounts_info())
        assert switcher._usage_store.entries(identity)["2"].fetched_at == before

    def test_refresh_fetches_and_stores_the_reading(self, refreshable, capsys):
        switcher, clock, identity = refreshable
        with _fetch_returning(NEW) as fetch:
            switcher.refresh_account_usage("2")
        assert fetch.call_count == 1
        entry = switcher._usage_store.entries(identity)["2"]
        assert entry.fetched_at == clock.now
        assert entry.last_good["five_hour"]["pct"] == 40.0
        assert "Refreshed Account-2 (account2@example.com)" in capsys.readouterr().out

    def test_a_read_within_the_minute_sends_nothing(self, refreshable):
        switcher, clock, identity = refreshable
        switcher._usage_store.record({"2": FetchRecord(usage=OLD)}, identity)
        clock.now += 30
        with _fetch_returning(NEW) as fetch, pytest.raises(ValidationError) as err:
            switcher.refresh_account_usage("2")
        assert fetch.call_count == 0
        assert "read 30s ago" in str(err.value)

    def test_a_running_backoff_sends_nothing(self, refreshable):
        switcher, _clock, identity = refreshable
        switcher._usage_store.record({"2": FetchRecord(error="timeout")}, identity)
        with _fetch_returning(NEW) as fetch, pytest.raises(ValidationError) as err:
            switcher.refresh_account_usage("2")
        assert fetch.call_count == 0
        assert "backing off" in str(err.value)

    def test_a_429_within_the_hour_sends_nothing(self, refreshable):
        switcher, clock, identity = refreshable
        store = switcher._usage_store
        store.record({"2": FetchRecord(error="http-429", retry_after_s=0.0)}, identity)
        clock.now += 400                        # past the backoff
        store.record({"2": FetchRecord(usage=OLD)}, identity)
        clock.now += 1400                       # 30 min after the 429
        with _fetch_returning(NEW) as fetch, pytest.raises(ValidationError) as err:
            switcher.refresh_account_usage("2")
        assert fetch.call_count == 0
        assert "rate-limited" in str(err.value)

    def test_a_failed_fetch_is_reported(self, refreshable, capsys):
        switcher, _clock, _identity = refreshable
        failed = oauth.UsageOutcome(None, error="http-403")
        with patch(
            "claude_swap.oauth.try_fetch_usage_for_account", return_value=failed
        ), pytest.raises(ValidationError) as err:
            switcher.refresh_account_usage("2")
        assert "the fetch failed" in str(err.value)
        assert "Not refreshed Account-2" in capsys.readouterr().out


class TestRefreshDispatch:
    def _run(self, argv):
        with patch("claude_swap.cli.ClaudeAccountSwitcher") as switcher_cls, \
             patch.object(sys, "argv", ["claude-swap", *argv]), \
             patch("os.geteuid", return_value=1000, create=True), \
             patch("claude_swap.update_check.check_for_update", return_value=None):
            cli.main()
        return switcher_cls.return_value

    def test_refresh_subcommand_forwards(self):
        switcher = self._run(["refresh", "17"])
        switcher.refresh_account_usage.assert_called_once_with("17")

    def test_refresh_by_email_forwards(self):
        switcher = self._run(["refresh", "user@example.com"])
        switcher.refresh_account_usage.assert_called_once_with("user@example.com")

    def test_refresh_without_target_errors(self):
        with patch.object(sys, "argv", ["claude-swap", "refresh"]):
            with pytest.raises(SystemExit) as excinfo:
                cli.main()
        assert excinfo.value.code == 2
