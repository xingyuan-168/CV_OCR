"""Mixed-model/Unicode and invalid-output checks using the actual x64 DLL.

Run after timing measurements, in the pinned reference environment. All synthetic
models and Unicode-path copies live in an owned temporary directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_yolo_compatibility import Native, digest, frame


def run(dll, cases):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    os.environ['CQ_AI_YOLO_CPU_THREADS'] = '1'
    native = Native(dll)
    handles, baselines = [], []
    labels = '\n'.join('类别' + str(i) for i in range(80)).encode('utf-8')
    selected = [c for c in cases if c['name'] in ('legacy_best', 'yolov5n-fp32-320',
                'yolo11n-fp32-320', 'yolo26n-fp32-320-dense', 'user_v26_end2end')]
    assert len(selected) == 5
    with tempfile.TemporaryDirectory(prefix='cq-yolo-多模型-') as temporary:
        root = Path(temporary)
        try:
            for case in selected:
                model = Path(case['model'])
                copied = root / ('模型-' + model.name)
                shutil.copy2(model, copied)
                bmp = frame(case['images'][0])[1]
                # The path entry is ACP on Windows; use the real legacy ABI.
                import ctypes as c
                h = c.c_int32(); assert native.dll.YOLO_Create(c.byref(h)) == 0
                handles.append(h.value)
                status = native.dll.YOLO_LoadModelFromPath(h.value, str(copied).encode('mbcs'),
                                                        None, 0, 2, 0, 1)
                assert status == 0, native.dll.AI_GetLastError()
                path_result = native.infer(h.value, bmp)
                assert native.dll.YOLO_Release(h.value) == 0; handles.pop()
                h = native.load(copied, 2, labels=labels, slots=1)
                handles.append(h)
                actual = json.loads(native.infer(h, bmp)); plain = json.loads(path_result)
                assert len(actual) == len(plain)
                for x, y in zip(actual, plain):
                    assert x['label'] == '类别' + str(x['class_id'])
                    assert {k:v for k,v in x.items() if k != 'label'} == {k:v for k,v in y.items() if k != 'label'}
                assert native.infer(h, bmp, conf=1) == '[]'
                baselines.append((bmp, native.infer(h, bmp)))

            def lane(index):
                for n in range(40):
                    k = (n + index) % len(handles)
                    bmp, expected = baselines[k]
                    assert native.infer(handles[k], bmp) == expected
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(lane, range(5)))
        finally:
            for h in handles: native.dll.YOLO_Release(h)

        # A real ONNX session produces invalid classes, NaN and threshold edges.
        def constant(rows, metadata, name):
            data = np.asarray(rows, dtype=np.float32)[None]
            graph = helper.make_graph([helper.make_node('Constant', [], ['output'],
                value=numpy_helper.from_array(data))], name,
                [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,320,320])],
                [helper.make_tensor_value_info('output', TensorProto.FLOAT, list(data.shape))])
            m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=9)
            helper.set_model_props(m, metadata)
            path = root / (name + '.onnx'); onnx.save(m, path); return path
        meta = {'task':'detect', 'end2end':'True', 'names':"{0:'one'}"}
        checked = []
        for name, row in [('class',[10,10,20,20,.9,1]), ('nan',[10,10,20,20,float('nan'),0])]:
            h = native.load(constant([row],meta,name), 2, slots=1)
            try:
                try: native.infer(h, baselines[0][0])
                except RuntimeError as error: assert 'invalid' in str(error).lower()
                else: raise AssertionError('Malformed output must fail: ' + name)
                checked.append(name)
            finally: native.dll.YOLO_Release(h)
        ambiguous = constant([[10,10,20,20,.9,0]], {'names':"{0:'one'}"}, 'ambiguous')
        try: native.load(ambiguous,2,slots=1)
        except RuntimeError as error: assert 'ambiguous' in str(error).lower()
        else: raise AssertionError('Ambiguous load must fail')
        checked.append('ambiguous')
        h = native.load(constant([[10,10,20,20,.5,0]],meta,'boundary'),2,slots=1)
        try:
            assert len(json.loads(native.infer(h,baselines[0][0],conf=.5))) == 1
            assert native.infer(h,baselines[0][0],conf=.5001) == '[]'
        finally: native.dll.YOLO_Release(h)
    return dict(passed=True, dll_sha256=digest(dll), models=[c['name'] for c in selected],
                chinese_paths=True, chinese_caller_labels=True, concurrent_calls=200,
                empty_results=True, threshold_equality=True, invalid_checks=checked)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    result=run(a.dll, json.loads(a.cases.read_text(encoding='utf-8')))
    a.report.write_text(json.dumps(result,ensure_ascii=True,indent=2),encoding='utf-8')
    print('Mixed model, Unicode, threshold and invalid-output integration passed')
