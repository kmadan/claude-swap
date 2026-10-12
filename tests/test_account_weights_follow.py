"""``autoswitch.accountWeights`` follows its accounts across remove, swap and move.

A weight describes the seat of the account in a slot. Slot numbers are reused
(``add`` numbers from the highest, ``swap`` and ``move`` renumber), so an entry
left on a freed or renumbered slot would weigh some other account's seat.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from claude_swap.settings import (
    load_settings,
    retarget_account_weights,
    set_setting,
    settings_path,
)
from claude_swap.switcher import ClaudeAccountSwitcher


class TestRetargetAccountWeights:
    def test_removed_account_drops_its_entry(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "6:5,19:5")
        assert retarget_account_weights(tmp_path, {"19": None}) == "6:5"
        assert load_settings(tmp_path).account_weights == "6:5"

    def test_moved_account_takes_its_entry_along(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "6:5,19:3")
        assert retarget_account_weights(tmp_path, {"6": "7"}) == "7:5,19:3"

    def test_swap_exchanges_entries_without_chaining(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "6:5,19:3")
        assert retarget_account_weights(tmp_path, {"6": "19", "19": "6"}) == "19:5,6:3"

    def test_last_entry_removed_unsets_the_key_and_keeps_others(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "19:5")
        set_setting(tmp_path, "autoswitch.threshold", "85")
        assert retarget_account_weights(tmp_path, {"19": None}) == ""
        raw = json.loads(settings_path(tmp_path).read_text())
        assert "accountWeights" not in raw["autoswitch"]
        assert raw["autoswitch"]["threshold"] == 85

    def test_last_key_removed_drops_the_section(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "19:5")
        assert retarget_account_weights(tmp_path, {"19": None}) == ""
        assert "autoswitch" not in json.loads(settings_path(tmp_path).read_text())

    def test_unrelated_slot_changes_nothing_and_writes_nothing(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "6:5")
        before = settings_path(tmp_path).read_text()
        with patch("claude_swap.settings.atomic_write_json") as write:
            assert retarget_account_weights(tmp_path, {"3": None}) is None
        write.assert_not_called()
        assert settings_path(tmp_path).read_text() == before

    def test_no_setting_no_file(self, tmp_path: Path):
        assert retarget_account_weights(tmp_path, {"3": None}) is None
        assert not settings_path(tmp_path).exists()

    def test_unparsable_entry_is_kept_as_written(self, tmp_path: Path):
        set_setting(tmp_path, "autoswitch.accountWeights", "6:5,junk")
        assert retarget_account_weights(tmp_path, {"6": None}) == "junk"


class TestSlotChangesCarryWeights:
    def _switcher(self, data: dict, weights: str) -> ClaudeAccountSwitcher:
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, data)
        set_setting(switcher.backup_dir, "autoswitch.accountWeights", weights)
        return switcher

    def _weights(self, switcher: ClaudeAccountSwitcher) -> str:
        return load_settings(switcher.backup_dir).account_weights

    def test_remove_drops_the_removed_slots_weight(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = self._switcher(sample_sequence_data, "2:5,1:2")
        with patch.object(switcher, "_delete_account_files"):
            switcher.remove_account("2", assume_yes=True)
        assert self._weights(switcher) == "1:2"

    def test_a_new_account_in_a_freed_slot_starts_unweighted(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """The 2026-10-12 case: slot 19's premium account was removed and a
        standard one later added under 19 inherited 19:5."""
        switcher = self._switcher(sample_sequence_data, "2:5")
        with patch.object(switcher, "_delete_account_files"):
            switcher.remove_account("2", assume_yes=True)
        assert self._weights(switcher) == ""

    def test_move_to_an_empty_slot_takes_the_weight_along(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = self._switcher(sample_sequence_data, "2:5")
        switcher.move_account("2", "5")
        assert self._weights(switcher) == "5:5"

    def test_swap_exchanges_the_weights(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = self._switcher(sample_sequence_data, "1:5")
        switcher.swap_accounts("1", "2")
        assert self._weights(switcher) == "2:5"

    def test_move_onto_an_occupied_slot_swaps_the_weights(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = self._switcher(sample_sequence_data, "1:5,2:3")
        switcher.move_account("1", "2")
        assert self._weights(switcher) == "2:5,1:3"

    def test_a_settings_write_failure_leaves_the_removal_in_place(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = self._switcher(sample_sequence_data, "2:5")
        with patch.object(switcher, "_delete_account_files"), patch(
            "claude_swap.switcher.retarget_account_weights", side_effect=OSError("disk full")
        ):
            switcher.remove_account("2", assume_yes=True)
        assert "2" not in switcher._get_sequence_data()["accounts"]
        assert self._weights(switcher) == "2:5"
