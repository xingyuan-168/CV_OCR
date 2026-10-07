from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from release_manifest import CORE_NAMES, current_manifest, safe_member, verify_archive, verify_cohort, verify_sources
from check_governance import verify_candidate


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

    def candidate(self):
        header = self.root / 'include/ai_engine.h'; header.parent.mkdir()
        header.write_text('#define AIENGINE_VERSION_MAJOR 0\n#define AIENGINE_VERSION_MINOR 14\n#define AIENGINE_VERSION_PATCH 8\n')
        protocol = self.root / 'src/worker_protocol.h'; protocol.parent.mkdir()
        protocol.write_text('constexpr int kVersion = 26;\n')
        sources = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in (header, protocol)}
        (self.root / 'release/current').mkdir(parents=True)
        (self.root / 'release/current.json').write_text(json.dumps(dict(schema=1, manifest='current/manifest.json')))
        (self.root / 'release/current/manifest.json').write_text(json.dumps(dict(schema=2, git_source_sha256=sources)))
        runtime = self.root / 'outpush/task/runtime'; runtime.mkdir(parents=True)
        fingerprints = {}
        for name in ('CQ_X86.dll', 'CQ_AI_worker.exe'):
            path = runtime / name; path.write_bytes(name.encode())
            fingerprints[name] = dict(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        probe = runtime.parent / 'loader.json'
        probe.write_text(json.dumps(dict(errors=0, public_exports=60, worker_protocol=26, version='CQ_X86/0.14.8',
            dll_sha256=fingerprints['CQ_X86.dll']['sha256'], worker_sha256=fingerprints['CQ_AI_worker.exe']['sha256'])))
        metadata = dict(project_version='0.14.8', functional_validation='passed', public_exports=60, worker_protocol=26,
            git_source_sha256=sources, candidate_runtime_directory=str(runtime), binary_fingerprints=fingerprints,
            functional_evidence=dict(file='task/loader.json', size=probe.stat().st_size, sha256=hashlib.sha256(probe.read_bytes()).hexdigest()))
        path = runtime.parent / 'metadata.json'
        path.write_text(json.dumps(metadata))
        return path, metadata

    def test_candidate_requires_actual_matching_loader_and_pair(self):
        path, metadata = self.candidate()
        verify_candidate(self.root, path, 'codex/test')
        missing = dict(metadata); missing.pop('functional_evidence')
        path.write_text(json.dumps(missing))
        with self.assertRaisesRegex(ValueError, 'paired-loader evidence'):
            verify_candidate(self.root, path, 'codex/test')
        path.write_text(json.dumps(metadata))
        (Path(metadata['candidate_runtime_directory']) / 'CQ_AI_worker.exe').write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            verify_candidate(self.root, path, 'codex/test')

    def test_candidate_cannot_bypass_source_snapshot_or_staging(self):
        path, metadata = self.candidate()
        with self.assertRaises(ValueError): verify_candidate(self.root, path, 'main')
        with self.assertRaises(ValueError): verify_candidate(self.root, self.root / 'input/metadata.json', 'codex/test')
        metadata['git_source_sha256'].pop('src/worker_protocol.h')
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify_candidate(self.root, path, 'codex/test')

    def test_candidate_loader_tamper_is_rejected(self):
        path, metadata = self.candidate()
        (self.root / 'outpush/task/loader.json').write_text('{}')
        with self.assertRaises(ValueError): verify_candidate(self.root, path, 'codex/test')

    def test_candidate_matching_hash_does_not_certify_failed_loader(self):
        path, metadata = self.candidate()
        probe_path = self.root / 'outpush/task/loader.json'
        probe = json.loads(probe_path.read_text()); probe['errors'] = 1
        probe_path.write_text(json.dumps(probe))
        metadata['functional_evidence'].update(size=probe_path.stat().st_size, sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest())
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'differs or failed'):
            verify_candidate(self.root, path, 'codex/test')


if __name__ == '__main__':
    unittest.main()
