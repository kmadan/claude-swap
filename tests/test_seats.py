"""Seat weights read from the stored account profiles (claude_swap.seats)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from claude_swap.seats import effective_weights, read_seat_tiers, tier_weight


def _store(root: Path, accounts: dict[str, tuple[str, object]]) -> Path:
    """A backup dir holding sequence.json and one stored profile per slot.

    ``accounts`` maps slot to (email, profile): a dict is written as JSON, a
    str verbatim, and None writes no file.
    """
    (root / "configs").mkdir(parents=True)
    sequence = {
        "accounts": {slot: {"email": email} for slot, (email, _p) in accounts.items()},
        "sequence": [int(slot) for slot in accounts],
    }
    (root / "sequence.json").write_text(json.dumps(sequence))
    for slot, (email, profile) in accounts.items():
        if profile is None:
            continue
        path = root / "configs" / f".claude-config-{slot}-{email}.json"
        path.write_text(profile if isinstance(profile, str) else json.dumps(profile))
    return root


def _profile(tier: str | None = None) -> dict:
    account = {"emailAddress": "x@example.com"}
    if tier is not None:
        account["userRateLimitTier"] = tier
    return {"oauthAccount": account}


class TestTierWeight:
    def test_a_multiplier_suffix_is_the_weight(self):
        assert tier_weight("default_claude_max_5x") == 5.0
        assert tier_weight("default_claude_max_20x") == 20.0

    def test_a_tier_without_a_multiplier_names_no_weight(self):
        assert tier_weight("default_raven") is None
        assert tier_weight("") is None
        assert tier_weight(None) is None

    def test_a_zero_multiplier_is_not_a_weight(self):
        assert tier_weight("default_claude_max_0x") is None


class TestReadSeatTiers:
    def test_reads_the_tier_of_every_profile_that_has_one(self, tmp_path):
        root = _store(tmp_path, {
            "1": ("big@example.com", _profile("default_claude_max_5x")),
            "2": ("std@example.com", _profile("default_raven")),
            "3": ("missing@example.com", None),
            "4": ("broken@example.com", "{not json"),
            "5": ("untiered@example.com", _profile()),
        })
        assert read_seat_tiers(root) == {
            "1": "default_claude_max_5x",
            "2": "default_raven",
        }

    def test_no_sequence_file_means_no_tiers(self, tmp_path):
        assert read_seat_tiers(tmp_path) == {}

    def test_a_changed_profile_is_read_again(self, tmp_path):
        root = _store(tmp_path, {"1": ("a@example.com", _profile("default_raven"))})
        cache: dict = {}
        assert read_seat_tiers(root, cache) == {"1": "default_raven"}
        path = root / "configs" / ".claude-config-1-a@example.com.json"
        path.write_text(json.dumps(_profile("default_claude_max_5x")))
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        assert read_seat_tiers(root, cache) == {"1": "default_claude_max_5x"}

    def test_an_unchanged_profile_comes_from_the_cache(self, tmp_path):
        root = _store(tmp_path, {"1": ("a@example.com", _profile("default_raven"))})
        cache: dict = {}
        read_seat_tiers(root, cache)
        path = root / "configs" / ".claude-config-1-a@example.com.json"
        cache[path] = (cache[path][0], "marked_9x")  # same mtime, marked value
        assert read_seat_tiers(root, cache) == {"1": "marked_9x"}


class TestEffectiveWeights:
    def test_a_premium_tier_weighs_its_slot(self):
        tiers = {"1": "default_claude_max_5x", "2": "default_raven"}
        assert effective_weights({}, tiers) == {"1": 5.0}

    def test_an_explicit_weight_wins_over_the_tier(self):
        tiers = {"1": "default_claude_max_5x"}
        assert effective_weights({"1": 2.0, "3": 3.0}, tiers) == {"1": 2.0, "3": 3.0}

    def test_an_explicit_one_overrides_a_premium_tier(self):
        assert effective_weights({"1": 1.0}, {"1": "default_claude_max_5x"}) == {"1": 1.0}
