#!/usr/bin/env python3
"""Clean-checkout import and collection gate (P0.1 — with unittest fallback).

Two checks that must pass on every node, including CPU-only shards where PyTorch or
pytest is absent:

1. every package/test/tool/kaggle module compiles (``compileall``), and
2. the whole test suite is *collectable* and runnable via ``pytest`` (or Python's
   built-in ``unittest`` loader when ``pytest`` is not installed), catching stray
   Markdown fences, bad imports and unguarded optional dependencies before they
   silently hide tests.
"""

from __future__ import annotations

import argparse
import compileall
import importlib.util
import pathlib
import subprocess
import sys
import unittest

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


def _has_pytest() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _collect_unittest() -> bool:
    sys.path.insert(0, str(PROJECT_ROOT))
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"))
    if loader.errors:
        for err in loader.errors:
            print(err, file=sys.stderr)
        print(f"COLLECT: FAILED ({len(loader.errors)} import error(s))", file=sys.stderr)
        return False
    count = suite.countTestCases()
    if count <= 0:
        print("COLLECT: FAILED (0 tests discovered)", file=sys.stderr)
        return False
    print(f"COLLECT: ok ({count} tests via unittest)")
    return True


def _run_unittest() -> bool:
    sys.path.insert(0, str(PROJECT_ROOT))
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(
            f"TESTS: FAILED (failures={len(result.failures)}, errors={len(result.errors)})",
            file=sys.stderr,
        )
        return False
    print(f"TESTS: ok ({result.testsRun} ran, {len(result.skipped)} skipped)")
    return True


def run_pytest(extra_args: list[str], label: str) -> bool:
    if not _has_pytest():
        if "--collect-only" in extra_args:
            return _collect_unittest()
        return _run_unittest()
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
        "--no-collect", action="store_true", help="skip test collection"
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
