import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_yolo_acceptance as acceptance


class LocalAcceptanceTest(unittest.TestCase):
    def run_summary(self, *, best_soak=True, p95=14, limit=15, fp32_gpu_growth=False, best_gpu_growth=False):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            def write(name, value):
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            report = dict(rounds=[dict(errors=0, elapsed_s=1800, lanes=[dict(p50_ms=10, p95_ms=p95)])])
            write("hardware.json", dict(business_windows_active=False, baseline_only=False))
            write("calibration-report.json", dict(p95_limit_ms=limit, best=dict(reports=[report]), auto_report=report,
                                                   auto_reports=[report, report]))
            write("acceptance-summary.json", dict(target_host=True, gpu=dict(gpu_soak=True, cache_hit=True, corruption_rebuilt=True)))
            write("optional-dependency-checks/summary.json", dict(passed=True))
            for name in ("first-build", "cache-hit", "corrupt-rebuild"):
                write("cache-" + name + ".detections.json", [])
            write("soak-tensorrt-fp32.json", report)
            if best_soak:
                write("soak.json", report)
            def gpu(path):
                grew = best_gpu_growth if path.name == "soak.gpu.csv" else fp32_gpu_growth
                return dict(available=True, growth_detected=grew)
            with patch.object(sys, "argv", ["summary", str(folder)]), \
                 patch.object(acceptance, "memory_summary", return_value=dict(continuous_growth_detected=False)), \
                 patch.object(acceptance, "gpu_memory_summary", side_effect=gpu), contextlib.redirect_stdout(io.StringIO()):
                acceptance.main()
            return json.loads((folder / "acceptance-summary.json").read_text(encoding="utf-8"))

    def test_bus_acceptance_does_not_certify_business_windows(self):
        result = self.run_summary()
        self.assertTrue(result["local_benchmark_accepted"])
        self.assertFalse(result["target_performance_accepted"])
        self.assertEqual(result["p95_limit_ms"], 15)

    def test_fp32_soak_cannot_replace_selected_candidate_soak(self):
        self.assertFalse(self.run_summary(best_soak=False)["local_benchmark_accepted"])

    def test_selected_gpu_growth_rejects_even_when_fp32_is_stable(self):
        self.assertFalse(self.run_summary(best_gpu_growth=True)["local_benchmark_accepted"])

    def test_fifteen_ms_limit_rejects_sixteen(self):
        self.assertFalse(self.run_summary(p95=16)["local_benchmark_accepted"])

    def test_user_revised_twenty_ms_limit_is_used(self):
        self.assertTrue(self.run_summary(p95=19.9, limit=20)["local_benchmark_accepted"])
        self.assertFalse(self.run_summary(p95=20.1, limit=20)["local_benchmark_accepted"])


if __name__ == "__main__":
    unittest.main()
