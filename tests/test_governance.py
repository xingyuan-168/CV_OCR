from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from release_manifest import CORE_NAMES, current_manifest, safe_member, verify_archive, verify_cohort, verify_sources


class ManifestGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def cohort(self):
        files = {}
        for name in CORE_NAMES:
            p = self.root / name
            p.write_bytes(name.encode('utf-8'))
            files[name] = dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        return files

    def test_path_escape_and_drive_rejected(self):
        for value in ('../outside', '/outside', 'C:/outside', 'C:outside', '..\\outside'):
            with self.subTest(path=value), self.assertRaises(ValueError):
                safe_member(self.root, value)

    def test_current_pointer_cannot_escape_release(self):
        (self.root / 'release').mkdir()
        (self.root / 'release/current.json').write_text(json.dumps(dict(schema=1, manifest='../outside.json')))
        with self.assertRaises(ValueError):
            current_manifest(self.root)

    def test_missing_or_extra_output_rejected(self):
        files = self.cohort()
        verify_cohort(self.root, files)
        (self.root / 'extra.txt').write_text('unexpected')
        with self.assertRaises(ValueError):
            verify_cohort(self.root, files)
        (self.root / 'extra.txt').unlink()
        (self.root / 'CQ_AI_worker.exe').unlink()
        with self.assertRaises(ValueError):
            verify_cohort(self.root, files)

    def test_binary_tamper_rejected_even_at_same_size(self):
        files = self.cohort()
        p = self.root / 'CQ_X86.dll'
        p.write_bytes(b'x' * p.stat().st_size)
        with self.assertRaises(ValueError):
            verify_cohort(self.root, files)

    def test_zip_traversal_rejected(self):
        p = self.root / 'bad.zip'
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('../escape', b'bad')
        expected = dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            verify_archive(p, expected)

    def test_source_line_endings_and_protocol_are_verified(self):
        header = self.root / 'include/ai_engine.h'; header.parent.mkdir()
        protocol = self.root / 'src/worker_protocol.h'; protocol.parent.mkdir()
        header.write_bytes(b'#define AIENGINE_VERSION_MAJOR 0\r\n#define AIENGINE_VERSION_MINOR 14\r\n#define AIENGINE_VERSION_PATCH 6\r\n')
        protocol.write_bytes(b'constexpr int kVersion = 26;\r\n')
        manifest = dict(project_version='0.14.6', worker_protocol=26, git_source_sha256={p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in (header, protocol)})
        verify_sources(self.root, manifest)
        manifest['worker_protocol'] = 25
        with self.assertRaises(ValueError):
            verify_sources(self.root, manifest)


if __name__ == '__main__':
    unittest.main()
