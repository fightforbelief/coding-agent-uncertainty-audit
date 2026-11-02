from census_metrics import census_flags


def _call(round_number: int, command: str | None, output: str | None, tool: str = "bash") -> dict:
    return {"round": round_number, "tool": tool, "command": command, "input": command or "", "output": output}


def test_high_confidence_failure_is_not_confused_with_ambiguous() -> None:
    calls = [
        _call(2, "python tests/test_app.py", "ok"),
        _call(4, "pytest -q", "FAILED tests/test_app.py::test_one\n===== 1 failed in 0.2s"),
        _call(12, "npm test", "Tests:       2 failed, 3 passed, 5 total"),
    ]
    flags = census_flags(calls)
    assert flags["v3_ran_tests_r5"] == 1
    assert flags["v3_ran_tests_amb_r5"] == 0
    assert flags["v3_n_test_runs_r5"] == 1
    assert flags["v3_test_fail_r5"] == 1
    assert flags["v3_test_fail_r15"] == 1
    assert flags["v3_n_test_runs_r15"] == 2
    assert flags["first_test_round"] == 4
    assert flags["v3_anyobs_fail_r5"] == 1


def test_ambiguous_only_and_loose_screen() -> None:
    calls = [_call(3, "python tests/test_app.py", "Traceback (most recent call last):\nValueError: x")]
    flags = census_flags(calls)
    assert flags["v3_ran_tests_r5"] == 0
    assert flags["v3_ran_tests_amb_r5"] == 1
    assert flags["v3_n_test_runs_r5"] == 0
    assert flags["v3_test_fail_r5"] == 0
    assert flags["loose_r5"] == 1
    assert flags["first_test_round"] is None


def test_build_failure_ambiguous_and_unknown_stay_separate() -> None:
    calls = [
        _call(1, "python tests/test_app.py", "ERROR collecting tests/test_app.py"),
        _call(2, "pytest -q", "ERROR collecting tests/test_app.py"),
        _call(3, "pytest -q", "not a recognized test summary"),
    ]
    flags = census_flags(calls)
    assert flags["v3_ran_tests_amb_r5"] == 0
    assert flags["v3_n_test_runs_r5"] == 2
    assert flags["v3_test_fail_r5"] == 0
    assert flags["v3_collect_build_fail_r5"] == 1
    assert flags["high_unknown_outputs_r5"] == 1
    assert flags["v3_anyobs_fail_r5"] == 0


def test_rounds_after_the_checkpoint_are_excluded() -> None:
    calls = [_call(6, "pytest -q", "===== 1 failed in 0.1s")]
    flags = census_flags(calls)
    assert flags["v3_test_fail_r5"] == 0
    assert flags["v3_test_fail_r15"] == 1
    assert flags["first_test_round"] == 6
