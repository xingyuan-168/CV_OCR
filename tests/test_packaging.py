from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import copy
import json
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_delivery import safe_output, ROOT, verified_reports
from register_release import register


class PackagingTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'outpush').mkdir(exist_ok=True)

    def test_base_and_external_outputs_protect_user_assets(self):
        for p in (ROOT/'output/bad.zip',ROOT/'input/bad.zip',ROOT/'release/bad.zip'):
            with self.subTest(path=p),self.assertRaises(ValueError): safe_output(p,external=True)
        self.assertEqual(safe_output(ROOT/'outpush/scratch/test.zip'),ROOT/'outpush/scratch/test.zip')
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(safe_output(Path(t)/'optional.zip',external=True),(Path(t)/'optional.zip').resolve())
            with self.assertRaises(ValueError):safe_output(Path(t)/'file.dll',external=True)

    def test_current_archives_can_be_packaged_without_sdk_or_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outpush',prefix='packaging-test-') as t:
            output=Path(t)/'delivery'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/package_delivery.py'),'--reuse-current','--out',str(output)],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr.decode('utf-8',errors='replace'))
            self.assertTrue((output/'manifest.json').exists())
            self.assertFalse((output/'core-stage').exists())
            again=subprocess.run([sys.executable,str(ROOT/'scripts/package_delivery.py'),'--reuse-current','--out',str(output)],capture_output=True)
            self.assertNotEqual(again.returncode,0)

    def test_registered_release_cannot_be_overwritten(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'manifest.json').write_text(json.dumps({'delivery_version':'v23.6'}))
            with self.assertRaisesRegex(ValueError,'immutable'):register(p,p/'unused.whl')

    def validated_reports(self, folder, p95=19, lanes=5):
        binaries = {'CQ_X86.dll': dict(sha256='a'), 'CQ_AI_worker.exe': dict(sha256='b')}
        metadata = dict(project_version='0.14.8', worker_protocol=26, local_benchmark_validation='passed',
                        local_benchmark_p95_limit_ms=20, local_benchmark_model_sha256='model', local_benchmark_workload='workload')
        base = dict(dll_sha256='a', worker_sha256='b', worker_protocol=26)
        loader = dict(base, errors=0, public_exports=60, version='CQ_X86/0.14.8')
        report = dict(base, api_version='CQ_X86/0.14.8', callers=5, warmup_per_lane=100, samples_per_lane=1000,
                      p95_limit_ms=20, model_sha256='model', workload='workload', confidence=.5,
                      runtime=dict(input_width=640, input_height=640),
                      rounds=[dict(errors=0, elapsed_s=15, lanes=[dict(count=1000, p50_ms=12, p95_ms=p95) for _ in range(lanes)]) for _ in range(3)])
        soak = copy.deepcopy(report); soak['rounds'] = soak['rounds'][:1]; soak['rounds'][0]['elapsed_s'] = 1800
        for name, value in (('delivery-build.json', loader), ('continuous.json', report), ('simultaneous.json', report),
                            ('soak-30min.json', soak), ('summary.json', dict(functional_passed=True, memory_growth_detected=False, local_benchmark_accepted=True))):
            (folder/name).write_text(json.dumps(value), encoding='utf-8')
        return SimpleNamespace(validation=folder), binaries, metadata

    def test_local_twenty_ms_release_requires_every_actual_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.validated_reports(Path(temporary))
            verified_reports(*args)
            with self.assertRaises(ValueError): verified_reports(*self.validated_reports(Path(temporary), p95=20.01))
            with self.assertRaises(ValueError): verified_reports(*self.validated_reports(Path(temporary), lanes=1))

    def test_local_release_rejects_a_different_threshold_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.validated_reports(Path(temporary))
            path = Path(temporary)/'continuous.json'; report = json.loads(path.read_text())
            report['p95_limit_ms'] = 30; path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): verified_reports(*args)
            report['p95_limit_ms'] = 20; report['model_sha256'] = 'other-model'; path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): verified_reports(*args)
