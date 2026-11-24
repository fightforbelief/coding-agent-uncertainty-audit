"""Exercise real runners in private temporary directories. No agent command is executed."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis_v2"))
from p8v3.testcmd import parse_test_output


PY_TEST = '''import os
import unittest
from pathlib import Path

class Example(unittest.TestCase):
    def setUp(self):
        if os.environ["P8_CONDITION"] == "setup_error":
            raise RuntimeError("fixture unavailable")

    @unittest.skipIf(os.environ["P8_CONDITION"] == "skip", "feature unavailable")
    def test_value(self):
        with Path("../entries").open("ab") as f:
            f.write(b"x")
        self.assertNotEqual(os.environ["P8_CONDITION"], "failure")
'''
GO_TEST = '''package sample
import ("os"; "testing")
func TestValue(t *testing.T) {
    f, err := os.OpenFile("../entries", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
    if err != nil { t.Fatal(err) }
    if _, err = f.WriteString("x"); err != nil { t.Fatal(err) }
    if err = f.Close(); err != nil { t.Fatal(err) }
}
'''


def run_suite(go):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        repo = base / "sample"
        repo.mkdir()
        (repo / "test_sample.py").write_text(PY_TEST)
        (repo / "sample_test.go").write_text(GO_TEST)
        (repo / "go.mod").write_text("module example.org/sample\n\ngo 1.22\n")
        counter = base / "entries"
        counter.write_bytes(b"")
        versions = {"go": subprocess.check_output([go,"version"], text=True).strip(),
                    "unittest": sys.version.split()[0],
                    "pytest": subprocess.check_output([sys.executable,"-m","pytest","--version"], text=True).strip()}
        jobs = []
        for runner in ("pytest","unittest"):
            args = [sys.executable,"-m",runner, "test_sample.py"]
            for condition, entries, exitcode in (("pass",1,0),("skip",0,0),("failure",1,1),("setup_error",0,1)):
                jobs.append((runner,condition,args,entries,exitcode))
        for name, extra, entries in (("cold",[],1),("warm",[],0),("forced",["-count=1"],1),
                                     ("no_match",["-run","^TestMissing$"],0)):
            jobs.append(("go",name,[go,"test","-v",*extra,"./..."],entries,0))
        results = []
        for runner, name, args, expected, exitcode in jobs:
            before = counter.stat().st_size
            env = {**os.environ, "P8_CONDITION":name, "GOCACHE":str(base/"cache"),
                   "GOTOOLCHAIN":"local", "GOWORK":"off", "GOPROXY":"off", "GOFLAGS":"",
                   "GOENV":"off", "PYTEST_ADDOPTS":"", "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1"}
            proc = subprocess.run(args, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=240)
            entries = counter.stat().st_size - before
            if entries != expected or proc.returncode != exitcode:
                raise RuntimeError(f"{runner}/{name}: witness={entries}, exit={proc.returncode}\n{proc.stdout[-1400:]}")
            cached = "(cached)" in proc.stdout
            if runner == "go" and cached != (name == "warm"):
                raise RuntimeError(f"Unexpected Go cache state in {name}")
            parsed = parse_test_output(proc.stdout)
            cmd = " ".join(["go" if runner == "go" else "python", *args[1:]])
            results.append(dict(runner=runner,version=versions[runner],condition=name,command=cmd,
                                exit_code=proc.returncode,body_entries=entries,old_parser=parsed.status,
                                cache_marker=int(cached),verbose_go_case_pass=int("--- PASS: TestValue" in proc.stdout)))
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", default="go")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = run_suite(args.go)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
