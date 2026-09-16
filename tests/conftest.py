"""Puts src on the import path so `python -m pytest tests` works."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
