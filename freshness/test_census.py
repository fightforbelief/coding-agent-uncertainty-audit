from census import package_counts


def test_mixed_package_records():
    counts = package_counts("cd repo && go test ./...", "ok  x/lib  (cached)\nok\tx/core\t0.012s\nok\tx/empty\t0.003s [no tests to run]\n? x/absent [no test files]")
    assert (counts["cached"], counts["timed"], counts["no_match"]) == (1, 1, 1)
    assert counts["tab_timed"] == 1 and counts["tab_cached"] == 0


def test_exclude_unattributable_text():
    assert package_counts("echo go test ./...", "ok x (cached)") is None
    assert package_counts("go test ./... | head", "ok x (cached)") is None
    assert package_counts("go test ./...", "example: ok x 0.001s\nok x 0.1s trailing text") == {}
