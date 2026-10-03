from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dependency_cache import cache
from repository_storage import copy_verified, info, verify_nvidia


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_copy_reuses_same_content_and_refuses_overwrite(self):
        a = self.root / 'a'; b = self.root / 'b'
        a.write_bytes(b'known')
        copy_verified(a, b); copy_verified(a, b)
        b.write_bytes(b'other')
        with self.assertRaises(ValueError): copy_verified(a, b)
        self.assertEqual(b.read_bytes(), b'other')

    def test_cache_checks_all_files_and_recipe(self):
        configs = self.root / 'configs'; configs.mkdir()
        lock = configs / 'dependencies.lock.json'
        spec = {'directory':'third_party/sdk', 'version':'1', 'required':['header.h']}
        lock.write_text(json.dumps({'groups':{'sdk':spec}}))
        header = self.root / 'third_party/sdk/header.h'; header.parent.mkdir(parents=True)
        header.write_bytes(b'valid')
        self.assertTrue(cache(self.root, 'sdk', True)['cache_hit'])
        header.write_bytes(b'other')
        with self.assertRaises(ValueError): cache(self.root, 'sdk')
        header.write_bytes(b'valid')
        spec['version']='2'; lock.write_text(json.dumps({'groups':{'sdk':spec}}))
        with self.assertRaises(ValueError): cache(self.root, 'sdk')

    def test_archive_member_hash_failure(self):
        z = self.root / 'nvidia.zip'
        bad = {'size':4, 'sha256':'0'*64}
        with zipfile.ZipFile(z,'w') as archive:
            archive.writestr('CQ_YOLO_TensorRT.dll',b'data')
            archive.writestr('manifest.json',json.dumps({'dependencies':{'files':{}}, 'module':bad}))
        with self.assertRaises(ValueError): verify_nvidia(z, info(z))

    @unittest.skipUnless(sys.platform == 'win32', 'PowerShell filesystem acceptance')
    def test_powershell_cleanup_boundaries(self):
        script = Path(__file__).with_name('cleanup_safety_test.ps1')
        for shell in filter(None,[shutil.which('powershell'),shutil.which('pwsh')]):
            with self.subTest(shell=shell), tempfile.TemporaryDirectory() as root:
                run = subprocess.run([shell,'-NoProfile','-File',str(script),'-Root',root],capture_output=True)
                self.assertEqual(run.returncode,0,run.stderr.decode('utf-8',errors='replace'))
