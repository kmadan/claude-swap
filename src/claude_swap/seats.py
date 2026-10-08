"""Seat sizes read from the stored account profiles.

A Team plan sells standard and premium seats, and a premium seat holds several
times the quota of a standard one. Claude Code records the seat in each
account's profile as ``oauthAccount.userRateLimitTier``, and the copy of that
profile cswap stores per slot (``configs/.claude-config-<slot>-<email>.json``)
keeps it: ``default_raven`` on a standard seat, ``default_claude_max_5x`` on a
premium one. A tier that ends in a multiplier names the seat's weight, which
``autoswitch.accountWeights`` otherwise has to be told by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_MULTIPLIER_RE = re.compile(r"_(\d+(?:\.\d+)?)x$")


def tier_weight(tier: str | None) -> float | None:
    """``default_claude_max_5x`` -> 5.0; None when the tier names no multiplier."""
    if not isinstance(tier, str):
        return None
    match = _MULTIPLIER_RE.search(tier.strip())
    if not match:
        return None
    weight = float(match.group(1))
    return weight if weight > 0 else None


def _stored_tier(path: Path, cache: dict | None) -> str | None:
    """``userRateLimitTier`` from one stored profile, cached on its mtime."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    if cache is not None and path in cache and cache[path][0] == mtime:
        return cache[path][1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    account = data.get("oauthAccount") if isinstance(data, dict) else None
    tier = account.get("userRateLimitTier") if isinstance(account, dict) else None
    tier = tier if isinstance(tier, str) and tier else None
    if cache is not None:
        cache[path] = (mtime, tier)
    return tier


def read_seat_tiers(backup_dir: Path, cache: dict | None = None) -> dict[str, str]:
    """``{slot: userRateLimitTier}`` for every managed account whose profile has one.

    Reads ``sequence.json`` for the slots and their emails, then each slot's
    stored profile. A slot whose profile is missing, unreadable or carries no
    tier is left out. ``cache`` (any dict the caller keeps) skips re-parsing a
    profile whose file has not changed, for callers that ask on every tick.
    """
    try:
        sequence = json.loads((backup_dir / "sequence.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    accounts = sequence.get("accounts") if isinstance(sequence, dict) else None
    if not isinstance(accounts, dict):
        return {}
    tiers: dict[str, str] = {}
    for slot, record in accounts.items():
        email = record.get("email") if isinstance(record, dict) else None
        if not isinstance(email, str) or not email:
            continue
        path = backup_dir / "configs" / f".claude-config-{slot}-{email}.json"
        tier = _stored_tier(path, cache)
        if tier:
            tiers[str(slot)] = tier
    return tiers


def effective_weights(explicit: dict[str, float], tiers: dict[str, str]) -> dict[str, float]:
    """Seat weights: each tier's multiplier, with ``explicit`` entries winning.

    ``explicit`` is ``autoswitch.accountWeights`` as parsed. A slot it names
    keeps that weight whatever its tier says, so the setting can still correct
    a tier or weigh an account the tier does not describe. A slot named by
    neither weighs 1 and is left out, as before.
    """
    weights: dict[str, float] = {}
    for slot, tier in tiers.items():
        weight = tier_weight(tier)
        if weight is not None and weight != 1.0:
            weights[slot] = weight
    weights.update(explicit)
    return weights
