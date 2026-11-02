import pytest
from p8v3.testcmd import detect_test_command, parse_test_output, loose_fail_screen

POS = [
    ("pytest tests/test_a.py -x", "pytest"),
    ("cd /testbed && python -m pytest tests/ -k foo 2>&1 | tail -50", "pytest"),
    ("PYTHONPATH=. timeout 300 python3 -m pytest -q", "pytest"),
    ("python -m unittest discover -s tests", "unittest"),
    ("python3.9 -m unittest tests.test_x", "unittest"),
    ("bash -c \"cd /repo && go test ./pkg/... -run TestFoo\"", "go"),
    ("cd /workspace/repo; go test -v ./...", "go"),
    ("cargo test --lib parser", "cargo"),
    ("npm test -- --runInBand", "npm"),
    ("npm run test:unit", "npm"),
    ("yarn test src/x.test.ts", "yarn"),
    ("yarn jest foo", "yarn"),
    ("pnpm test", "pnpm"),
    ("npx jest src/a.test.js", "jest"),
    ("npx --yes vitest run", "vitest"),
    ("./node_modules/.bin/mocha test/a.js", "mocha"),
    ("mvn -q test -Dtest=FooTest", "maven"),
    ("./gradlew :core:test --tests Foo", "gradle"),
    ("make test", "make"),
    ("cd build && ctest --output-on-failure", "ctest"),
    ("bundle exec rspec spec/foo_spec.rb", "rspec"),
    ("vendor/bin/phpunit tests/FooTest.php", "phpunit"),
    ("poetry run pytest", "pytest"),
    ("uv run pytest -x", "pytest"),
    ("tox -e py39", "tox"),
    ("python tests/runtests.py queries", "runtests_script"),
    ("./runtests.py admin_views", "runtests_script"),
    ("python manage.py test app", "django"),
    ("dotnet test", "dotnet"),
    ("env CI=1 npm test", "npm"),
    ("(cd sub && pytest)", "pytest"),
    ("cd /repo && \\\n  pytest -q", "pytest"),
]

NEG = [
    "cat tests/test_a.py",
    "grep -rn pytest setup.cfg",
    "pip install pytest",
    "echo 'run pytest later'",
    "find . -name '*test*'",
    "python reproduce.py",
    "python -c 'import pytest; print(pytest.__version__)'",
    "ls tests/",
    "go build ./...",
    "cargo build",
    "npm install",
    "vim tests/test_x.py",
    "sed -n '1,40p' tests/test_x.py",
    "make",
    "git diff",
    "",
    None,
]


@pytest.mark.parametrize("cmd,runner", POS)
def test_detect_positive(cmd, runner):
    r = detect_test_command(cmd)
    assert r is not None, cmd
    assert r.runner == runner, (cmd, r)
    assert r.confidence == "high"


@pytest.mark.parametrize("cmd", NEG)
def test_detect_negative(cmd):
    assert detect_test_command(cmd) is None, cmd


def test_ambiguous_test_file():
    r = detect_test_command("python tests/test_issue.py")
    assert r and r.confidence == "ambiguous"
    r = detect_test_command("python test_repro.py")
    assert r and r.confidence == "ambiguous"


OUT = {
    "pytest_fail": ("""collected 3 items

tests/test_a.py .F.                                                     [100%]

=================================== FAILURES ===================================
____________________________________ test_b ____________________________________
E       AssertionError: assert 1 == 2
=========================== short test summary info ============================
FAILED tests/test_a.py::test_b - AssertionError: assert 1 == 2
========================= 1 failed, 2 passed in 0.12s ==========================
""", "fail"),
    "pytest_pass": ("============================== 5 passed in 0.31s ===============================\n", "pass"),
    "pytest_collect": ("""==================================== ERRORS ====================================
______________________ ERROR collecting tests/test_a.py ________________________
ImportError while importing test module '/r/tests/test_a.py'.
=========================== short test summary info ============================
ERROR tests/test_a.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.20s ===============================
""", "collect_or_build_fail"),
    "unittest_fail": ("""F.
======================================================================
FAIL: test_x (tests.test_m.T)
----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)
""", "fail"),
    "unittest_ok": ("..\n----------------------------------------------------------------------\nRan 2 tests in 0.001s\n\nOK\n", "pass"),
    "unittest_err": ("Ran 1 test in 0.0s\n\nFAILED (errors=1)\n", "error_only"),
    "go_fail": ("=== RUN   TestFoo\n--- FAIL: TestFoo (0.00s)\n    foo_test.go:10: bad\nFAIL\nFAIL\tgithub.com/x/y\t0.012s\n", "fail"),
    "go_pkgfail_only": ("FAIL\tgithub.com/x/y\t0.012s\n", "fail"),
    "go_ok": ("ok  \tgithub.com/x/y\t0.010s\n", "pass"),
    "go_build": ("# github.com/x/y\n./a.go:3:2: undefined: Foo\nFAIL\tgithub.com/x/y [build failed]\n", "collect_or_build_fail"),
    "cargo_fail": ("test tests::a ... FAILED\n\ntest result: FAILED. 1 passed; 1 failed; 0 ignored\n", "fail"),
    "cargo_ok": ("test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured\n", "pass"),
    "cargo_compile": ("error[E0425]: cannot find value `x` in this scope\nerror: could not compile `foo`\n", "collect_or_build_fail"),
    "jest_fail": ("Tests:       1 failed, 3 passed, 4 total\n", "fail"),
    "jest_pass": ("Tests:       4 passed, 4 total\n", "pass"),
    "mocha_fail": ("  3 passing (20ms)\n  1 failing\n", "fail"),
    "rspec_fail": ("Finished in 0.1 seconds\n3 examples, 1 failure\n", "fail"),
    "rspec_ok": ("3 examples, 0 failures\n", "pass"),
    "phpunit_fail": ("FAILURES!\nTests: 5, Assertions: 9, Failures: 1.\n", "fail"),
    "maven_fail": ("Tests run: 10, Failures: 2, Errors: 0, Skipped: 0\n", "fail"),
    "maven_zero": ("Tests run: 10, Failures: 0, Errors: 0, Skipped: 0\n", "unknown"),
    "ctest_fail": ("90% tests passed, 1 tests failed out of 10\n", "fail"),
    "repro_traceback": ("Traceback (most recent call last):\n  File \"r.py\", line 1\nValueError: x\n", "unknown"),
    "empty": ("", "unknown"),
}


@pytest.mark.parametrize("name", list(OUT))
def test_parse_output(name):
    text, status = OUT[name]
    assert parse_test_output(text).status == status, name


def test_loose_screen():
    assert loose_fail_screen("Traceback (most recent call last):")
    assert loose_fail_screen("1 failed")
    assert not loose_fail_screen("all good, 5 passed")
    assert not loose_fail_screen("")
