"""
p8v3.testcmd — detect test-command executions and parse test-runner output.

Independent of parser v2. Pure functions, no data dependencies, so they can be
unit-tested offline and imported by analysis_v2 scripts.

Public API
----------
detect_test_command(cmd: str) -> TestCmd | None
    Given the raw shell command text of ONE tool call, return which test runner
    it invokes (or None). Handles `cd x && pytest`, env-var prefixes, `timeout`,
    `bash -c '...'`, `poetry run`, `npx`, `bundle exec`, etc.

parse_test_output(text: str) -> TestOutcome
    Given the observation text returned for that call, classify it into
    test-case failure / all passed / collection-or-build failure / unknown.

loose_fail_screen(text: str) -> bool
    Deliberately over-inclusive screen used ONLY to define sampling strata
    for the recall audit (never as a model feature).
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Command detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCmd:
    runner: str            # e.g. "pytest", "go", "jest"
    confidence: str        # "high" | "ambiguous"
    segment: str           # the sub-command that matched


_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
# wrappers whose remaining argv is the real command
_WRAPPERS_NOARG = {"time", "sudo", "nohup", "env", "command", "exec", "xvfb-run", "stdbuf", "nice"}
_WRAPPERS_ONEARG = {"timeout"}           # timeout 60 pytest ...
_RUN_PREFIXES = [                         # multi-token prefixes to strip
    ("poetry", "run"), ("uv", "run"), ("pipenv", "run"), ("pdm", "run"),
    ("hatch", "run"), ("bundle", "exec"), ("pnpm", "exec"), ("yarn", "exec"),
]
_PY = re.compile(r"^(?:.*/)?python(?:\d+(?:\.\d+)?)?$")
_TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test|tests?/[^ ]*)\.py$")

_NPM_LIKE = {"npm", "yarn", "pnpm", "bun"}
_DIRECT = {
    "pytest": "pytest", "py.test": "pytest", "tox": "tox", "nox": "nox",
    "jest": "jest", "vitest": "vitest", "mocha": "mocha", "ava": "ava",
    "rspec": "rspec", "phpunit": "phpunit", "ctest": "ctest",
    "nosetests": "nose", "trial": "trial",
}


def _strip_wrappers(toks: list[str]) -> list[str]:
    changed = True
    while toks and changed:
        changed = False
        while toks and _ENV_ASSIGN.match(toks[0]):
            toks = toks[1:]; changed = True
        if not toks:
            break
        base = toks[0].rsplit("/", 1)[-1]
        if base in _WRAPPERS_NOARG:
            toks = toks[1:]
            # skip wrapper flags like `env -i`, `timeout -s KILL`
            while toks and toks[0].startswith("-"):
                toks = toks[1:]
            changed = True; continue
        if base in _WRAPPERS_ONEARG:
            toks = toks[1:]
            while toks and toks[0].startswith("-"):
                toks = toks[2:] if toks[0] in ("-s", "-k", "--signal", "--kill-after") else toks[1:]
            if toks:
                toks = toks[1:]          # the duration
            changed = True; continue
        for pre in _RUN_PREFIXES:
            if tuple(toks[:len(pre)]) == pre:
                toks = toks[len(pre):]; changed = True; break
        if toks and base == "conda" and len(toks) > 1 and toks[1] == "run":
            toks = toks[2:]
            while toks and toks[0].startswith("-"):
                toks = toks[2:] if toks[0] in ("-n", "--name", "-p", "--prefix") else toks[1:]
            changed = True
        if toks and toks[0].rsplit("/", 1)[-1] in ("npx", "pnpx", "bunx"):
            toks = toks[1:]
            while toks and toks[0].startswith("-"):
                toks = toks[1:]
            changed = True
    return toks


def _classify(toks: list[str]) -> tuple[str, str] | None:
    if not toks:
        return None
    t0 = toks[0]
    base = t0.rsplit("/", 1)[-1]
    rest = toks[1:]

    if base in _DIRECT:
        return _DIRECT[base], "high"

    if _PY.match(base):
        # python -m pytest / unittest / tox / nose
        if len(rest) >= 2 and rest[0] == "-m":
            mod = rest[1]
            if mod in ("pytest", "py.test"):
                return "pytest", "high"
            if mod.startswith("unittest"):
                return "unittest", "high"
            if mod in ("tox", "nox", "nose", "nose2", "twisted.trial"):
                return mod.split(".")[0], "high"
            if mod == "django" and len(rest) > 2 and rest[2] == "test":
                return "django", "high"
        if rest and rest[0] == "-c":
            return None
        args = [a for a in rest if not a.startswith("-")]
        if args:
            script = args[0]
            sb = script.rsplit("/", 1)[-1]
            if sb in ("runtests.py", "run_tests.py", "runtest.py"):
                return "runtests_script", "high"
            if sb == "manage.py" and len(args) > 1 and args[1] == "test":
                return "django", "high"
            if sb == "setup.py" and len(args) > 1 and args[1] == "test":
                return "setup_py", "high"
            if _TEST_FILE.search(script):
                # could be a project test file OR an agent-written repro script
                return "python_test_file", "ambiguous"
        return None

    if base in ("runtests.py", "run_tests.py"):
        return "runtests_script", "high"

    if base == "go" and rest[:1] == ["test"]:
        return "go", "high"
    if base == "cargo" and rest[:1] in (["test"], ["nextest"], ["t"]):
        return "cargo", "high"
    if base in _NPM_LIKE:
        if not rest:
            return None
        a = rest[0]
        if a in ("test", "t", "tst"):
            return base, "high"
        if a == "run" and len(rest) > 1 and re.match(r"^(test|tests|unit|spec)\b", rest[1]):
            return base, "high"
        if base != "npm" and re.match(r"^(test|jest|vitest|mocha)\b", a):
            return base, "high"
        return None
    if base in ("mvn", "mvnw", "./mvnw") or base == "mvnw":
        if any(a in ("test", "verify", "surefire:test") or a.startswith("-Dtest=") for a in rest):
            return "maven", "high"
        return None
    if base in ("gradle", "gradlew"):
        if any(re.search(r"(^|:)(test|check|[A-Za-z]*Test)$", a) for a in rest):
            return "gradle", "high"
        return None
    if base in ("make", "gmake"):
        if any(re.match(r"^(test|tests|check|unittest|test-\S+|tests?_\S+)$", a) for a in rest):
            return "make", "high"
        return None
    if base == "rake" and any(a in ("test", "spec") or a.startswith("test:") for a in rest):
        return "rake", "high"
    if base == "dotnet" and rest[:1] == ["test"]:
        return "dotnet", "high"
    if base == "sbt" and any(re.search(r"\btest(Only|Quick)?\b", a) for a in rest):
        return "sbt", "high"
    if base == "mix" and rest[:1] == ["test"]:
        return "mix", "high"
    if base == "bazel" and rest[:1] == ["test"]:
        return "bazel", "high"
    if base == "deno" and rest[:1] == ["test"]:
        return "deno", "high"
    if base == "node" and "--test" in rest:
        return "node_test", "high"
    if base == "swift" and rest[:1] == ["test"]:
        return "swift", "high"
    return None


def _segments(cmd: str):
    """Split on &&, ||, ;, |, newline — but not inside quotes or $(...)."""
    buf, out = [], []
    q = None
    depth = 0
    i, s = 0, cmd.strip()
    while i < len(s):
        c = s[i]
        if q:
            buf.append(c)
            if c == "\\" and q == '"' and i + 1 < len(s):
                buf.append(s[i + 1]); i += 2; continue
            if c == q:
                q = None
            i += 1; continue
        if c in ("'", '"'):
            q = c; buf.append(c); i += 1; continue
        if c == "\\" and i + 1 < len(s):
            if s[i + 1] == "\n":       # line continuation
                buf.append(" "); i += 2; continue
            buf.append(c); buf.append(s[i + 1]); i += 2; continue
        if s.startswith("$(", i):
            depth += 1; buf.append("$("); i += 2; continue
        if c == ")" and depth:
            depth -= 1; buf.append(c); i += 1; continue
        if depth == 0:
            two = s[i:i + 2]
            if two in ("&&", "||"):
                out.append("".join(buf)); buf = []; i += 2; continue
            if c in (";", "\n", "|", "&"):
                out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    out.append("".join(buf))
    for seg in out:
        seg = seg.strip()
        while seg.startswith("(") and seg.endswith(")"):
            seg = seg[1:-1].strip()
        seg = seg.lstrip("(").rstrip(")").strip() if seg.count("(") != seg.count(")") else seg
        if seg:
            yield seg


def detect_test_command(cmd: str | None, _depth: int = 0) -> TestCmd | None:
    """Return the first test-runner invocation found in a shell command, else None."""
    if not cmd or _depth > 3:
        return None
    best: TestCmd | None = None
    for seg in _segments(cmd):
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        toks = _strip_wrappers(toks)
        if not toks:
            continue
        # bash -c '...' / sh -c "..."
        if toks[0].rsplit("/", 1)[-1] in ("bash", "sh", "zsh") and "-c" in toks:
            i = toks.index("-c")
            if i + 1 < len(toks):
                inner = detect_test_command(toks[i + 1], _depth + 1)
                if inner and (best is None or inner.confidence == "high"):
                    best = inner
                    if best.confidence == "high":
                        return best
            continue
        r = _classify(toks)
        if r:
            hit = TestCmd(runner=r[0], confidence=r[1], segment=seg)
            if hit.confidence == "high":
                return hit
            best = best or hit
    return best


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

@dataclass
class TestOutcome:
    test_failed: bool = False         # >=1 test case failed (assertion-level)
    test_passed_summary: bool = False # a summary line shows tests ran and none failed
    collect_or_build_fail: bool = False
    test_error_only: bool = False     # e.g. pytest "N error", unittest errors= without failures=
    frameworks: set = field(default_factory=set)

    @property
    def status(self) -> str:
        if self.test_failed:
            return "fail"
        if self.collect_or_build_fail:
            return "collect_or_build_fail"
        if self.test_error_only:
            return "error_only"
        if self.test_passed_summary:
            return "pass"
        return "unknown"


M = re.MULTILINE
_FAIL_PATTERNS = [
    ("pytest", re.compile(r"^=+ .*\b[1-9]\d* failed\b.* in [\d.]+s", M)),
    ("pytest", re.compile(r"^FAILED \S+::\S+", M)),
    ("pytest", re.compile(r"^\S+\.py::\S+ FAILED\b", M)),
    ("unittest", re.compile(r"^FAILED \((?:[^)]*\b)?failures=[1-9]", M)),
    ("go", re.compile(r"^\s*--- FAIL: \S+", M)),
    ("cargo", re.compile(r"^test result: FAILED\.", M)),
    ("cargo", re.compile(r"^test \S+ \.\.\. FAILED$", M)),
    ("jest", re.compile(r"^Tests:\s+(?:.*?, )?[1-9]\d* failed", M)),
    ("mocha", re.compile(r"^\s*[1-9]\d* failing\s*$", M)),
    ("rspec", re.compile(r"\b\d+ examples?, [1-9]\d* failures?\b")),
    ("phpunit", re.compile(r"^Tests: \d+, Assertions: \d+, (?:Errors: \d+, )?Failures: [1-9]", M)),
    ("maven", re.compile(r"Tests run: \d+, Failures: [1-9]\d*")),
    ("gradle", re.compile(r"\b\d+ tests? completed, [1-9]\d* failed\b")),
    ("ctest", re.compile(r"\d+% tests passed, [1-9]\d* tests? failed out of \d+")),
    ("dotnet", re.compile(r"Failed!\s+-\s+Failed:\s+[1-9]")),
]
_PASS_PATTERNS = [
    ("pytest", re.compile(r"^=+ (?:(?!failed).)*\b\d+ passed\b(?:(?!failed).)* in [\d.]+s", M)),
    ("unittest", re.compile(r"^Ran \d+ tests? in [\d.]+s\s*\n\s*\n?OK\b", M)),
    ("go", re.compile(r"^ok\s+\S+\s+(?:[\d.]+s|\(cached\))", M)),
    ("cargo", re.compile(r"^test result: ok\. [1-9]\d* passed", M)),
    ("jest", re.compile(r"^Tests:\s+(?:(?!failed).)*\d+ passed", M)),
    ("mocha", re.compile(r"^\s*[1-9]\d* passing", M)),
    ("rspec", re.compile(r"\b[1-9]\d* examples?, 0 failures\b")),
    ("phpunit", re.compile(r"^OK \(\d+ tests?", M)),
    ("ctest", re.compile(r"100% tests passed")),
    ("dotnet", re.compile(r"Passed!\s+-\s+Failed:\s+0")),
]
_COLLECT_BUILD = [
    ("pytest", re.compile(r"^ERROR collecting\b|^=+ ERRORS =+$|errors? during collection|^INTERNALERROR>", M)),
    ("pytest", re.compile(r"^ImportError while (?:loading conftest|importing test module)", M)),
    ("go", re.compile(r"^FAIL\s+\S+\s+\[(?:build|setup) failed\]", M)),
    ("go", re.compile(r"^# \S+\n.*\.go:\d+:\d+: ", M)),
    ("cargo", re.compile(r"^error(?:\[E\d+\])?: could not compile|^error\[E\d{4}\]", M)),
    ("jest", re.compile(r"Test suite failed to run")),
    ("maven", re.compile(r"COMPILATION ERROR|BUILD FAILURE")),
    ("gradle", re.compile(r"Compilation failed|^FAILURE: Build failed", M)),
    ("ts", re.compile(r"error TS\d{4}:")),
]
_ERROR_ONLY = [
    ("pytest", re.compile(r"^=+ (?:(?!failed).)*\b[1-9]\d* errors?\b(?:(?!failed).)* in [\d.]+s", M)),
    ("unittest", re.compile(r"^FAILED \(errors=[1-9]\d*\)", M)),
]


def parse_test_output(text: str | None) -> TestOutcome:
    out = TestOutcome()
    if not text:
        return out
    for fw, pat in _FAIL_PATTERNS:
        if pat.search(text):
            out.test_failed = True; out.frameworks.add(fw)
    for fw, pat in _PASS_PATTERNS:
        if pat.search(text):
            out.test_passed_summary = True; out.frameworks.add(fw)
    for fw, pat in _COLLECT_BUILD:
        if pat.search(text):
            out.collect_or_build_fail = True; out.frameworks.add(fw)
    if not out.test_failed:
        for fw, pat in _ERROR_ONLY:
            if pat.search(text):
                out.test_error_only = True; out.frameworks.add(fw)
    # go: a package-level "FAIL\tpkg 0.1s" without a --- FAIL line is still a test failure
    if not out.test_failed and re.search(r"^FAIL\s+\S+\s+[\d.]+s\s*$", text, M):
        out.test_failed = True; out.frameworks.add("go")
    return out


_LOOSE = re.compile(
    r"\bFAIL(?:ED|URES?)?\b|\b[1-9]\d* (?:failed|failing|failures?)\b|AssertionError|"
    r"assert(?:ion)? failed|Traceback \(most recent call last\)|\bpanic:|"
    r"\berror(?:\[E\d+\])?:|\bERROR\b",
)


def loose_fail_screen(text: str | None) -> bool:
    """Over-inclusive screen. Sampling-stratum definition only; never a model feature."""
    return bool(text) and bool(_LOOSE.search(text))
