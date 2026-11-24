"""Check verbose Go cache replay on named tests from a local Go distribution."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PACKAGES = ("encoding/base32", "encoding/base64", "hash/adler32", "hash/crc32",
            "hash/fnv", "math/bits", "sort", "unicode/utf8")
TEST = re.compile(r"^func (Test[A-Z]\w*)\(\w+ \*testing\.T\)\s*\{", re.M)
PACKAGE = re.compile(r"^package \w+\b", re.M)


def run(go):
    go = shutil.which(go) or str(Path(go).resolve())
    src = Path(subprocess.check_output([go, "env", "GOROOT"], text=True).strip()) / "src"
    version = subprocess.check_output([go, "version"], text=True).strip()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        env = {**os.environ, "GOCACHE": str(base/"cache"), "GOENV": "off", "GOFLAGS": "",
               "GOPROXY": "off", "GOWORK": "off", "GOTOOLCHAIN": "local"}
        for pkg in PACKAGES:
            counter = base / "entries"
            counter.write_bytes(b"")
            # Choose by file name then declaration order. Ignore platform-tagged files.
            candidates = sorted((src/pkg).glob("*_test.go"))
            for file in candidates:
                original = file.read_text(encoding="utf-8")
                if "//go:build" in original or not (test := TEST.search(original)):
                    continue
                break
            else:
                raise ValueError(f"No eligible test in {pkg}")
            name = test[1]
            # Use an external file so verbose output replay cannot create an entry.
            probe = ('\n{ f, e := p8os.OpenFile(' + json.dumps(str(counter)) +
                     ', p8os.O_APPEND|p8os.O_WRONLY, 0600); if e != nil { panic(e) }; '
                     'if _, e = f.WriteString("x"); e != nil { panic(e) }; '
                     'if e = f.Close(); e != nil { panic(e) } }\n')
            modified = original[:test.end()] + probe + original[test.end():]
            modified = PACKAGE.sub(lambda m: m[0] + '\nimport p8os "os"', modified, count=1)
            substitute = base / "instrumented.go"
            substitute.write_text(modified, encoding="utf-8")
            overlay = base / "overlay.json"
            overlay.write_text(json.dumps({"Replace": {str(file): str(substitute)}}), encoding="utf-8")
            for condition, extra in (("cold", []), ("warm", []), ("forced", ["-count=1"])):
                before = counter.stat().st_size
                cmd = [go, "test", "-overlay", str(overlay), "-v", "-run", "^"+name+"$", *extra, pkg]
                proc = subprocess.run(cmd, cwd=base, env=env, capture_output=True, text=True,
                                      errors="replace", timeout=240)
                if proc.returncode:
                    raise RuntimeError(f"{pkg}/{name}: {proc.stdout[-600:]} {proc.stderr[-600:]}")
                entries = counter.stat().st_size - before
                cached = bool(re.search(r"^ok\s+\S+\s+\(cached\)", proc.stdout, re.M))
                named_pass = bool(re.search(r"^--- PASS: "+name+r" \(", proc.stdout, re.M))
                results.append(dict(package=pkg, test=name, version=version, condition=condition,
                                    body_entries=entries, cache_marker=int(cached),
                                    verbose_case_pass=int(named_pass), exit_code=proc.returncode))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", default="go")
    ap.add_argument("--output", type=Path, required=True)
    opts = ap.parse_args()
    rows = run(opts.go)
    with opts.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
