import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from promote_verified_delivery import promote_transaction
from release_manifest import CORE_NAMES,digest


class TransactionTests(unittest.TestCase):
    def fixture(self,root):
        (root/"output").mkdir();(root/"release").mkdir();(root/"outpush/new").mkdir(parents=True)
        pointer=b'{"schema":1,"manifest":"v23.8/manifest.json"}\n'
        (root/"release/current.json").write_bytes(pointer)
        for name in CORE_NAMES:
            (root/"output"/name).write_bytes(b"old")
            (root/"outpush/new"/name).write_bytes(b"new")
        files={p.name:dict(size=3,sha256=digest(p)) for p in (root/"outpush/new").iterdir()}
        return pointer,files

    def test_postcheck_failure_restores_both_output_and_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pointer,files=self.fixture(root)
            def reject():
                self.assertEqual((root/"output/CQ_X86.dll").read_bytes(),b"new")
                self.assertIn("v23.9",(root/"release/current.json").read_text())
                raise RuntimeError("Finish rejection")
            with self.assertRaisesRegex(RuntimeError,"Finish rejection"):
                promote_transaction(root,root/"outpush/new",files,"v23.9",reject)
            self.assertEqual((root/"release/current.json").read_bytes(),pointer)
            for name in CORE_NAMES:self.assertEqual((root/"output"/name).read_bytes(),b"old")
            self.assertEqual(len(list((root/"outpush").glob("rejected-v23.9-*"))),1)

    def test_success_retains_original_cohort_for_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,files=self.fixture(root)
            result=promote_transaction(root,root/"outpush/new",files,"v23.9",lambda:None)
            self.assertTrue(result["verified"])
            for name in CORE_NAMES:
                self.assertEqual((root/"output"/name).read_bytes(),b"new")
                self.assertEqual((Path(result["rollback"])/name).read_bytes(),b"old")

    def test_unexpected_output_is_preserved_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pointer,files=self.fixture(root)
            (root/"output/user.txt").write_bytes(b"user")
            with self.assertRaisesRegex(ValueError,"unexpected"):
                promote_transaction(root,root/"outpush/new",files,"v23.9",lambda:None)
            self.assertEqual((root/"release/current.json").read_bytes(),pointer)
            self.assertEqual((root/"output/user.txt").read_bytes(),b"user")


if __name__=="__main__":unittest.main()
