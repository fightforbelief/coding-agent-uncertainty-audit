"""Queue rules. Enrichment is not a prevalence sample."""

from p8audit.execution_signals import classify_execution
from p8audit.review_queue import (
    context_excerpt,
    is_hard_negative,
    per_group_caps,
    signal_families,
    stratify,
)
import random


def test_exit_zero_test_failure_is_discordant_and_not_a_pass() -> None:
    text = "FAILED tests/a.py::t - AssertionError\n[The command completed with exit code 0.]\n"
    signals = classify_execution(text)
    families = signal_families(signals)
    assert "explicit_test_failure" in families
    assert "discordant_exit0_test_failure" in families
    assert "explicit_test_pass" not in families
    assert "tool_exit_nonzero" not in families


def test_interrupt_family_excludes_ordinary_nonzero_exit() -> None:
    signals = classify_execution("[The command completed with exit code 130. CTRL+C was sent.]\n")
    families = signal_families(signals)
    assert families == ["interrupt_or_timeout"]


def test_missed_format_is_not_a_frozen_pass_family() -> None:
    signals = classify_execution("Test Suites: 8 passed, 8 total\n")
    assert signal_families(signals) == ["missed_test_format"]


def test_hard_negative_requires_a_cue_and_no_positive_family() -> None:
    plain = classify_execution("total 2\nREADME.md\n")
    assert is_hard_negative("total 2\nREADME.md\n", plain) is False
    weak = classify_execution("copied the error handler\n")
    assert is_hard_negative("copied the error handler\n", weak) is True
    missed = classify_execution("Ran all test suites.\n")
    assert is_hard_negative("Ran all test suites.\n", missed) is False


def test_excerpt_keeps_context_around_the_hit() -> None:
    text = "\n".join([f"line {index}" for index in range(30)] + ["FAILED tests/a.py::t"])
    excerpt = context_excerpt(text, ["pytest_failed_line"])
    assert "[HEAD]" in excerpt
    assert "[TAIL]" in excerpt
    assert "FAILED tests/a.py::t" in excerpt
    assert "line 0" in excerpt


def test_two_populated_groups_are_topped_up_to_fifty() -> None:
    assert per_group_caps([10000, 10000, 0, 0]) == [25, 25, 0, 0]


def test_short_group_spills_the_remainder() -> None:
    assert per_group_caps([10, 10000, 0, 0]) == [10, 40, 0, 0]


def test_family_below_fifty_keeps_every_candidate() -> None:
    assert per_group_caps([4, 0, 7, 0]) == [4, 0, 7, 0]


def test_stratify_spreads_languages() -> None:
    items = []
    for language, count in (("python", 10), ("go", 10), ("rust", 1)):
        for index in range(count):
            items.append({"language": language, "index": index})
    chosen = stratify(items, 5, random.Random(20261026))
    assert len(chosen) == 5
    assert "rust" in {item["language"] for item in chosen}
