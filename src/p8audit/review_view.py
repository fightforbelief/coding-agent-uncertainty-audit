"""Short views of tool observations for manual review."""

from __future__ import annotations


def review_view(text: str) -> dict:
    lines = (text or "").splitlines()
    interesting = []
    needles = (
        "Traceback",
        "Error:",
        "Exception:",
        "Panic:",
        "exit code",
        "Exit code",
        "returncode",
        "FAILED",
        "FAILURES",
        "--- FAIL:",
        "test result:",
        "\nFAIL",
    )
    # The FAIL needle is applied to the raw line, not with a leading newline.
    line_needles = tuple(item for item in needles if item != "\nFAIL")
    for line in lines:
        stripped = line.lstrip()
        if any(needle in line for needle in line_needles) or stripped == "FAIL" or stripped.startswith("FAIL\t") or stripped.startswith("FAIL "):
            interesting.append(line[:240])
        if len(interesting) >= 8:
            break
    return {
        "n_lines": len(lines),
        "head": "\n".join(lines[:8])[:1200],
        "tail": "\n".join(lines[-6:])[:800],
        "signal_lines": interesting,
    }
