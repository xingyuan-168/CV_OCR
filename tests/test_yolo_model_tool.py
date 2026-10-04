import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('yolo_model_tool',ROOT/'tools/yolo_model_tool.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


class ModelToolTests(unittest.TestCase):
    def test_protected_and_existing_destinations(self):
        for path in (ROOT/'input/new.onnx',ROOT/'output/new.onnx',ROOT/'release/new.onnx',ROOT/'input/best.onnx'):
            with self.assertRaises(ValueError):tool.safe_destination(ROOT/'input/best.onnx',path)
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'exists.onnx';file.write_bytes(b'keep')
            with self.assertRaises(ValueError):tool.safe_destination(ROOT/'input/best.onnx',file)
            self.assertEqual(file.read_bytes(),b'keep')
            external=Path(folder)/'another-project'; (external/'input/models').mkdir(parents=True)
            source=external/'input/models/source.onnx'; source.write_bytes(b'keep')
            with self.assertRaises(ValueError):tool.safe_destination(source,external/'input/new.onnx')
            (external/'.git').mkdir()
            with self.assertRaises(ValueError):tool.safe_destination(source,external/'output/new.onnx')
            self.assertEqual(tool.safe_destination(source,external/'prepared/new.onnx'),external/'prepared/new.onnx')

    def test_metadata_only_copy_and_reference(self):
        try:
            import onnx
            from onnx import helper,TensorProto
            inspector=tool.inspector_path(os.environ.get('CQ_AI_YOLO_INSPECTOR'))
            import onnxruntime  # noqa: F401
        except (ImportError,ValueError) as exc:self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.onnx';copy=Path(folder)/'copy.onnx'
            data=helper.make_tensor('d',TensorProto.FLOAT,[1,2,6],[1,1,3,3,.8,0,4,4,6,6,.7,0])
            graph=helper.make_graph([helper.make_node('Constant',[],['detections'],value=data)],'tiny',
                [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,8,8])],
                [helper.make_tensor_value_info('detections',TensorProto.FLOAT,[1,2,6])])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8)
            entry=model.metadata_props.add();entry.key,entry.value='names',"{0:'object'}"
            source.write_bytes(model.SerializeToString());before=source.read_bytes()
            self.assertFalse(tool.inspect(source,inspector)['compatible'])
            report=tool.prepare(source,copy,inspector,'end2end',1,'BNC')
            self.assertTrue(report['compatible']);self.assertEqual(report['tensor_comparisons'],3)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(onnx.load(source).graph.SerializeToString(),onnx.load(copy).graph.SerializeToString())
            with self.assertRaises(ValueError):tool.prepare(source,copy,inspector,'end2end',1,'BNC')
            invalid=Path(folder)/'invalid.onnx'
            with self.assertRaises(ValueError):tool.prepare(source,invalid,inspector,'dense',2,'BCN')
            self.assertFalse(invalid.exists())


if __name__=='__main__':unittest.main()
