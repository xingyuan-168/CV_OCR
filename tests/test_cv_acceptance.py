from __future__ import annotations
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_cv_acceptance import INPUT_HASHES, REQUIRED_CASES, LEGACY_CASES, required, verify_candidate_cv, verify_timing
import package_delivery


class CvAcceptanceTests(unittest.TestCase):
    def report(self):
        r = dict(count=1000, errors=0, p50_ms=2., p95_ms=4.8, p99_ms=5.1, max_ms=8., over5_fraction=.03)
        return dict(dll_sha256="abc", architecture="x86", test_hooks=False, input_sha256=INPUT_HASHES,
                    warmup=64, samples=1000, rounds=3, limit_ms=5., accepted_5ms=True,
                    cases=[dict(name=n, rounds=[copy.deepcopy(r) for _ in range(3)]) for n in sorted(REQUIRED_CASES)])

    def test_valid(self):
        verify_timing(self.report(), "abc", "x86")

    def test_each_case_and_round_is_required(self):
        r = self.report(); r["cases"][-1]["rounds"][2]["p95_ms"] = 5.01
        with self.assertRaises(ValueError): verify_timing(r, "abc", "x86")
        r = self.report(); r["cases"].pop()
        with self.assertRaises(ValueError): verify_timing(r, "abc", "x86")

    def test_no_relaxed_or_instrumented_acceptance(self):
        for key,value in [("limit_ms",20.),("test_hooks",True),("samples",100),("accepted_5ms",False)]:
            r=self.report();r[key]=value
            with self.assertRaises(ValueError):verify_timing(r,"abc","x86")

    def test_binary_architecture_and_inputs_are_bound(self):
        with self.assertRaises(ValueError): verify_timing(self.report(),"different","x86")
        with self.assertRaises(ValueError): verify_timing(self.report(),"abc","x64")
        r=self.report();r["input_sha256"]={}
        with self.assertRaises(ValueError):verify_timing(r,"abc","x86")

    def test_old_versions_remain_immutable(self):
        self.assertFalse(required({"project_version":"0.14.8"}))
        self.assertTrue(required({"project_version":"0.14.9"}))
        self.assertEqual(verify_candidate_cv({"project_version":"0.14.8"},Path("unused"),{}),{})

    def test_legacy_slow_scenario_cannot_be_omitted_or_averaged(self):
        r=self.report()
        r["cases"]=[dict(name=n,rounds=copy.deepcopy(r["cases"][0]["rounds"])) for n in LEGACY_CASES]
        verify_timing(r,"abc","x86",legacy=True)
        r["cases"][0]["rounds"][1]["p95_ms"]=5.01
        with self.assertRaises(ValueError):verify_timing(r,"abc","x86",legacy=True)
        r["cases"].pop(0)
        with self.assertRaises(ValueError):verify_timing(r,"abc","x86",legacy=True)

    def test_failed_candidate_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError,"has not passed"):
            verify_candidate_cv(dict(project_version="0.14.9",cv_transparent_validation="failed"),Path("unused"),{})

    def test_stress_evidence_is_hashed_and_bound_to_production_dll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {}
            def write(key, report):
                data = json.dumps(report).encode()
                name = key + ".json"
                (root / name).write_bytes(data)
                evidence[key] = dict(file=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
            for arch in ("x86", "x64"):
                timing = self.report(); timing["architecture"] = arch
                write("timing_" + arch, timing)
                legacy = copy.deepcopy(timing)
                legacy["cases"] = [dict(name=n,rounds=copy.deepcopy(timing["cases"][0]["rounds"])) for n in LEGACY_CASES]
                write("legacy_" + arch, legacy)
                for threads in (1, 4, 16):
                    write(f"stress_{arch}_{threads}", dict(dll_sha256="abc", architecture=arch,
                          threads=threads, test_hooks=False, elapsed_s=600, errors=0, calls=100,
                          changed_cases=7, template_release_passed=True, memory_growth_detected=False))
            metadata = dict(project_version="0.14.9", cv_transparent_validation="passed",
                            cv_transparent_p95_limit_ms=5, cv_transparent_evidence=evidence,
                            cv_x64_sha256="abc", cv_independent_reference_validation="passed")
            binaries = {"CQ_X86.dll": dict(sha256="abc")}
            verify_candidate_cv(metadata, root, binaries)
            stress_path = root / evidence["stress_x64_16"]["file"]
            stress_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                verify_candidate_cv(metadata, root, binaries)

    def test_all_package_paths_reject_failed_cv_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dll = root / "CQ_AI_x64.dll"; dll.write_bytes(b"candidate")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(dict(project_version="0.14.9", delivery_version="v23.9",
                cv_transparent_validation="failed", cv_transparent_p95_limit_ms=5,
                cv_x64_sha256=hashlib.sha256(b"candidate").hexdigest(), binary_fingerprints={})))
            variants = [[], ["--python-only"], ["--nvidia-only", "--nvidia-deps", str(root),
                                               "--nvidia-zip", str(root / "nvidia.zip")]]
            for variant in variants:
                with self.subTest(variant=variant), patch.object(package_delivery, "current_manifest", return_value=(metadata, {"delivery_version":"v23.8"})), \
                        patch.object(package_delivery, "source_metadata"), patch("release_manifest.verify_sources"), \
                        patch.object(sys, "argv", ["package_delivery", "--metadata", str(metadata),
                                                  "--x64", str(root), "--dumpbin", "unused", "--out", str(root / "package"), *variant]):
                    with self.assertRaisesRegex(ValueError, "has not passed"):
                        package_delivery.main()
                    self.assertFalse((root / "package").exists())


if __name__ == "__main__": unittest.main()
