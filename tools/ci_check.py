#!/usr/bin/env python3
"""Clean-checkout import gate (P0.1).

Two checks that must pass on every node, including CPU-only shards where PyTorch is
absent:

1. every package/test/tool/kaggle module compiles (``compileall``), and
2. the whole test suite is *collectable* (``pytest --collect-only``), which catches
   stray Markdown fences, bad imports and unguarded optional dependencies before
   they silently hide tests.

Usage::

    python3 tools/ci_check.py            # compile + collect
    python3 tools/ci_check.py --no-collect
    python3 tools/ci_check.py --run      # also execute the suite
"""

from __future__ import annotations

import argparse
import compileall
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORIES = ("maya250m", "tests", "tools", "kaggle")
ENTRY_POINTS = ("run_maya250m.py",)


def compile_everything() -> bool:
    ok = True
    for directory in PACKAGE_DIRECTORIES:
        target = PROJECT_ROOT / directory
        if not target.exists():
            print(f"MISSING   {directory}", file=sys.stderr)
            ok = False
            continue
        # ``quiet=True`` suppresses per-file output; it does not suppress errors.
        if not compileall.compile_dir(str(target), quiet=1, force=True):
            print(f"COMPILE   {directory}: FAILED", file=sys.stderr)
            ok = False
        else:
            print(f"COMPILE   {directory}: ok")
    for entry in ENTRY_POINTS:
        target = PROJECT_ROOT / entry
        if not target.exists():
            print(f"MISSING   {entry}", file=sys.stderr)
            ok = False
            continue
        if not compileall.compile_file(str(target), quiet=1, force=True):
            print(f"COMPILE   {entry}: FAILED", file=sys.stderr)
            ok = False
        else:
            print(f"COMPILE   {entry}: ok")
    return ok


def run_pytest(extra_args: list[str], label: str) -> bool:
    command = [sys.executable, "-m", "pytest", "tests/"] + extra_args
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
    except FileNotFoundError:
        print(f"{label}: pytest is not installed", file=sys.stderr)
        return False
    if completed.returncode != 0:
        print(f"{label}: FAILED (exit {completed.returncode})", file=sys.stderr)
        return False
    print(f"{label}: ok")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-collect", action="store_true", help="skip pytest --collect-only"
    )
    parser.add_argument("--run", action="store_true", help="also execute the suite")
    args = parser.parse_args(argv)

    ok = compile_everything()
    if not args.no_collect:
        ok = run_pytest(["--collect-only", "-q"], "COLLECT") and ok
    if args.run:
        ok = run_pytest(["-q"], "TESTS") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
