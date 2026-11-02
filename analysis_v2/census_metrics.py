"""Census flags from adapter calls. Rules live in p8v3.testcmd and are not copied.

Metric meanings, for one trajectory and one K in {5, 15}. The window is
``round <= K``. These are not task outcomes and not parser v2 labels.

- ``v3_ran_tests_rK``: 1 if any call in the window has a high-confidence test
  command. This is command detection, not evidence that tests passed.
- ``v3_ran_tests_amb_rK``: 1 only when the window has an ambiguous test command
  and no high-confidence hit. Ambiguous is not high-confidence.
- ``v3_n_test_runs_rK``: number of high-confidence test-command calls in the
  window. It counts calls, not trajectories and not completed suites.
- ``v3_test_fail_rK``: 1 if any of those high-confidence calls has
  ``parse_test_output(...).test_failed``. A build failure alone does not set it.
- ``v3_collect_build_fail_rK``: 1 if any high-confidence call has
  ``collect_or_build_fail``. This is not an assertion failure.
- ``v3_anyobs_fail_rK``: 1 if any call output in the window, test command or
  not, has ``test_failed``.
- ``loose_rK``: 1 if any call output in the window matches ``loose_fail_screen``.
  This screen is not a test-failure decision.
- ``high_runners_rK``: ``|``-joined runner names for high-confidence hits.
- ``high_unknown_outputs_rK``: how many of those high-confidence outputs have
  status ``unknown``. Unknown is not success and not failure.
- ``first_test_round``: round of the first high-confidence hit in the whole
  trajectory, or missing.
"""

from __future__ import annotations

from typing import Any

from p8v3.testcmd import detect_test_command, loose_fail_screen, parse_test_output

CHECKPOINTS = (5, 15)


def _within(calls: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return [call for call in calls if int(call["round"]) <= k]


def census_flags(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the v3 census columns for one trajectory."""
    values: dict[str, Any] = {}
    for k in CHECKPOINTS:
        window = _within(calls, k)
        high = []
        ambiguous = []
        for call in window:
            hit = detect_test_command(call.get("command"))
            if hit is None:
                continue
            if hit.confidence == "high":
                high.append((call, hit))
            elif hit.confidence == "ambiguous":
                ambiguous.append((call, hit))
        values[f"v3_ran_tests_r{k}"] = int(bool(high))
        values[f"v3_ran_tests_amb_r{k}"] = int(bool(ambiguous) and not high)
        values[f"v3_n_test_runs_r{k}"] = len(high)
        values[f"v3_test_fail_r{k}"] = int(
            any(parse_test_output(call.get("output")).test_failed for call, _hit in high)
        )
        values[f"v3_collect_build_fail_r{k}"] = int(
            any(parse_test_output(call.get("output")).collect_or_build_fail for call, _hit in high)
        )
        values[f"v3_anyobs_fail_r{k}"] = int(
            any(parse_test_output(call.get("output")).test_failed for call in window)
        )
        values[f"loose_r{k}"] = int(any(loose_fail_screen(call.get("output")) for call in window))
        values[f"high_runners_r{k}"] = "|".join(hit.runner for _call, hit in high)
        values[f"high_unknown_outputs_r{k}"] = sum(
            parse_test_output(call.get("output")).status == "unknown" for call, _hit in high
        )
    first = next(
        (
            int(call["round"])
            for call in calls
            if (hit := detect_test_command(call.get("command"))) is not None and hit.confidence == "high"
        ),
        None,
    )
    values["first_test_round"] = first
    return values
