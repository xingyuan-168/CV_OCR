import unittest
import sys
import ctypes as C
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from ai_engine import AIEngineError, OCRCoordResult, OCRTextResult, _parse_compact_points


class CompactResultTests(unittest.TestCase):
    def test_cv_points(self):
        self.assertEqual(
            _parse_compact_points("0,364,28|1,302,18", "x", "y"),
            [{"id": 0, "x": 364, "y": 28}, {"id": 1, "x": 302, "y": 18}],
        )

    def test_ocr_gaps_repeats_and_negative_values(self):
        self.assertEqual(
            _parse_compact_points("1,-231,417|1,-25,-75|4,0,0", "cx", "cy"),
            [
                {"id": 1, "cx": -231, "cy": 417},
                {"id": 1, "cx": -25, "cy": -75},
                {"id": 4, "cx": 0, "cy": 0},
            ],
        )

    def test_empty(self):
        self.assertEqual(_parse_compact_points("", "x", "y"), [])

    def test_malformed(self):
        for value in ("|0,1,2", "0,1,2|", "0,1", "0,1,2,3", "x,1,2", "-1,1,2"):
            with self.subTest(value=value):
                with self.assertRaises(AIEngineError):
                    _parse_compact_points(value, "x", "y")

    def test_ocr_compat_result_layouts_match_packed_c_abi(self):
        self.assertEqual(C.sizeof(OCRTextResult), 28)
        self.assertEqual(C.sizeof(OCRCoordResult), 20)


if __name__ == "__main__":
    unittest.main()
