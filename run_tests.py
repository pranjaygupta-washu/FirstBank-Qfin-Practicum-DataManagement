"""
Test runner that works without pytest installed.

    python run_tests.py

If pytest is available, `python -m pytest tests -q` also works thanks to
tests/conftest.py putting src on the import path.
"""
import inspect
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_contract as T  # noqa: E402

tests = [(n, f) for n, f in vars(T).items()
         if n.startswith("test_") and callable(f)]

passed = failed = 0
for name, fn in tests:
    try:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
        print(f"PASS  {name}")
        passed += 1
    except Exception as exc:
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        failed += 1

print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
sys.exit(1 if failed else 0)
