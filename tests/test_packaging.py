from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_delivery import safe_output, ROOT
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
