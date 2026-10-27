"""Pure sampling rules for the Gate 0.5 marker-review queue.

The enriched queue is for measurement validation. Its mix of positives is not
a prevalence, and it is not a recall sample.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable

from p8audit.execution_signals import ExecutionSignals

POSITIVE_FAMILIES = (
    "explicit_exception",
    "explicit_test_failure",
    "explicit_test_pass",
    "tool_exit_nonzero",
    "interrupt_or_timeout",
    "missed_test_format",
    "discordant_exit0_test_failure",
)
OTHER_FAMILIES = ("hard_negative", "random_observation")
ALL_FAMILIES = POSITIVE_FAMILIES + OTHER_FAMILIES

_WEAK_CUE = re.compile(
    r"(?i)\b(error|failed|failure|passed|traceback|exception|exit code|assert)\b"
)


def signal_families(signals: ExecutionSignals) -> list[str]:
    """Families for which this observation is a candidate positive."""
    found: list[str] = []
    if signals.explicit_exception:
        found.append("explicit_exception")
    if signals.explicit_test_failure:
        found.append("explicit_test_failure")
    if signals.explicit_test_pass:
        found.append("explicit_test_pass")
    if signals.tool_exit_nonzero:
        found.append("tool_exit_nonzero")
    if signals.interrupt_or_timeout:
        found.append("interrupt_or_timeout")
    if (
        signals.exploratory_test_format
        and not signals.explicit_test_failure
        and not signals.explicit_test_pass
    ):
        found.append("missed_test_format")
    exit_zero = signals.scaffold_exit_codes == (0,)
    if signals.explicit_test_failure and exit_zero and not signals.tool_exit_nonzero:
        found.append("discordant_exit0_test_failure")
    return found


def is_hard_negative(text: str, signals: ExecutionSignals) -> bool:
    """Suspicious text that none of the positive families claim.

    Known missed test formats are their own family. This residual queue is
    for cues that the exploratory patterns also did not catch.
    """
    if signal_families(signals):
        return False
    return bool(_WEAK_CUE.search(text or ""))


def context_excerpt(text: str, reasons: Iterable[str], limit: int = 4000) -> str:
    """Head, match lines with neighbors, and tail. Not only the hit string."""
    raw = text or ""
    lines = raw.splitlines()
    head = "\n".join(lines[:15])[:1800]
    tail = "\n".join(lines[-10:])[:1200]
    interesting: list[str] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if any(token and token in line for token in (
            "Traceback",
            "Error:",
            "Exception:",
            "exit code",
            "Exit code",
            "FAILED",
            "FAIL",
            "passed",
            "test result:",
            "Test Suites",
            "Test Files",
            "Tests:",
            "CTRL+C",
            "timed out",
        )):
            interesting.append(f"L{number}: {stripped[:220]}")
        if len(interesting) >= 12:
            break
    body = "\n".join(
        [
            "[HEAD]",
            head,
            "[MATCH_LINES]",
            "\n".join(interesting) if interesting else "(none)",
            "[TAIL]",
            tail,
            f"[META] n_lines={len(lines)} n_chars={len(raw)} reasons={','.join(reasons)}",
        ]
    )
    if len(body) > limit:
        body = body[: limit - 20] + "\n[TRUNCATED]"
    return body


class Reservoir:
    """Deterministic reservoir of compact records."""

    def __init__(self, limit: int, rng: random.Random) -> None:
        self.limit = limit
        self.rng = rng
        self.seen = 0
        self.items: list[dict] = []

    def consider(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(item)
            return
        draw = self.rng.randrange(self.seen)
        if draw < self.limit:
            self.items[draw] = item


def per_group_caps(
    seen_by_group: list[int],
    *,
    minimum_total: int = 50,
    base_per_group: int = 16,
) -> list[int]:
    """How many rows to queue from each group.

    Populated groups start at ``base_per_group``. If the family has at least
    ``minimum_total`` candidates and the base caps do not reach that total,
    extra rows are taken from groups that still have unused candidates.
    A family with fewer than ``minimum_total`` candidates contributes all of them.
    """
    if any(value < 0 for value in seen_by_group):
        raise ValueError("seen counts must be non-negative")
    if minimum_total < 1 or base_per_group < 1:
        raise ValueError("caps must be positive")
    total_seen = sum(seen_by_group)
    if total_seen == 0:
        return [0 for _ in seen_by_group]
    if total_seen < minimum_total:
        return list(seen_by_group)
    caps = [min(seen, base_per_group) if seen else 0 for seen in seen_by_group]
    target = max(sum(caps), minimum_total)
    remaining = target - sum(caps)
    while remaining > 0:
        progressed = False
        for index, seen in enumerate(seen_by_group):
            if remaining == 0:
                break
            spare = seen - caps[index]
            if spare <= 0:
                continue
            caps[index] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return caps


def stratify(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Round-robin across language, then config is already a separate call."""
    if len(items) <= k:
        return list(items)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_language[str(item.get("language") or "unknown")].append(item)
    languages = sorted(by_language)
    rng.shuffle(languages)
    for language in languages:
        rng.shuffle(by_language[language])
    chosen: list[dict] = []
    while len(chosen) < k and any(by_language.values()):
        for language in languages:
            bucket = by_language[language]
            if bucket and len(chosen) < k:
                chosen.append(bucket.pop())
    return chosen
