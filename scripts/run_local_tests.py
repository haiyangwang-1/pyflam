"""Run the local unit-test layer that does not require MATLAB references."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


LOCAL_TEST_MODULES = [
    "test_core",
    "test_sparse_core",
    "test_utilities",
    "test_geom",
    "test_dense_algorithms",
    "test_mf",
    "test_hifie",
    "test_hifde",
    "test_matlab_parity_utils",
]


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module_name in LOCAL_TEST_MODULES:
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(module_name)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
