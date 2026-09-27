from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_release.py"
SPEC = importlib.util.spec_from_file_location("verify_release", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class ReleaseTest(unittest.TestCase):
    def test_public_release(self) -> None:
        result = VERIFY.run_checks(ROOT)
        self.assertEqual(result["measurement_floor"]["rows"], 96)
        self.assertEqual(result["measurement_floor"]["finite_estimate"], 43)
        self.assertEqual(result["sensitive_pattern_findings"], [])


if __name__ == "__main__":
    unittest.main()
