"""Record the remote environment without printing secrets."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 - environment probe must record the failure
        return f"ERROR {type(exc).__name__}: {exc}"
    text = (completed.stdout or "") + (completed.stderr or "")
    return text.strip()


def main() -> None:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "datasets",
        "huggingface_hub",
        "pyarrow",
        "matplotlib",
        "pytest",
        "yaml",
        "torch",
    ):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            packages[name] = f"MISSING {type(exc).__name__}"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_header": "\n".join(_run(["nvidia-smi"]).splitlines()[:8]),
        "git": _run(["git", "--version"]),
        "pip": _run([sys.executable, "-m", "pip", "--version"]),
        "conda": shutil.which("conda"),
        "uv": shutil.which("uv"),
    }
    out = ROOT / "artifacts" / "manifests" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
