"""Inspect detection ONNX files or produce verified metadata-only copies.

inspect needs only Python and the bundled native inspector. prepare additionally
uses ONNX/numpy/ONNX Runtime offline; none are DLL deployment dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspector_path(explicit=None):
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).with_name('ai_engine_yolo_model_inspect.exe'),
        ROOT / 'build/release-x64/Release/ai_engine_yolo_model_inspect.exe',
        ROOT / 'build/ci-x64/Release/ai_engine_yolo_model_inspect.exe',
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError('The matching native model inspector is required; pass --inspector or use the tools package')


def inspect(model, inspector):
    result = subprocess.run([str(inspector), str(Path(model).resolve())], capture_output=True, text=True, encoding='utf-8')
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError('Native inspector did not return JSON: ' + result.stderr) from exc
    if result.returncode not in (0, 2) or (result.returncode == 0) != report['compatible']:
        raise ValueError('Native inspector failed: ' + result.stderr)
    report.update(model=str(Path(model).resolve()), sha256=sha256(model))
    return report


def safe_destination(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.exists():
        raise ValueError('Prepared copies must use a new destination; original/existing files are never overwritten')
    protected_paths = {(base / name).resolve() for base in (ROOT, Path.cwd())
                       for name in ('input', 'output', 'release')}
    # Portable tool packages can sit outside the source repository. Preserve the
    # source's input tree and any enclosing Git repository as well.
    for parent in source.parents:
        if parent.name.casefold() == 'input': protected_paths.add(parent)
        if (parent / '.git').exists():
            protected_paths.update((parent / name).resolve() for name in ('input', 'output', 'release'))
            break
    for protected in protected_paths:
        if output == protected or output.is_relative_to(protected):
            raise ValueError('Prepared models cannot be written into protected ' + str(protected))
    if output.suffix.lower() != '.onnx':
        raise ValueError('Prepared destination must end in .onnx')
    return output


def prepare(source, output, inspector, format_name=None, classes=None, layout=None, input_size=640):
    import numpy as np
    import onnx
    import onnxruntime as ort
    source = Path(source).resolve()
    output = safe_destination(source, output)
    initial = inspect(source, inspector)
    if initial['compatible']:
        format_name = format_name or initial['format']
        classes = classes or initial['classes']
        layout = layout or initial['layout']
    if format_name is None or classes is None:
        raise ValueError('Ambiguous models require --format and --classes; ' + initial.get('error', ''))
    if format_name not in ('dense', 'objectness', 'end2end') or not 1 <= classes <= 65531:
        raise ValueError('Invalid format or class count')
    layout = layout or ('BCN' if format_name == 'dense' else 'BNC')
    if layout not in ('BCN', 'BNC') or (format_name == 'end2end' and layout != 'BNC'):
        raise ValueError('Invalid output layout')
    model = onnx.load(source, load_external_data=False)
    # Preserve all graph/weight/opset fields; change only our metadata namespace.
    original_body = onnx.ModelProto(); original_body.CopyFrom(model); original_body.ClearField('metadata_props')
    props = {item.key: item.value for item in model.metadata_props}
    if len(props) != len(model.metadata_props):
        raise ValueError('Duplicate source metadata must be corrected at export')
    props.update({'cq.yolo.schema': '1', 'cq.yolo.format': format_name, 'cq.yolo.classes': str(classes), 'cq.yolo.layout': layout})
    model.ClearField('metadata_props')
    for key, value in props.items():
        item = model.metadata_props.add(); item.key, item.value = key, value
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix='cq-model-', suffix='.onnx', dir=output.parent, delete=False) as stream:
            temp = Path(stream.name); stream.write(model.SerializeToString())
        checked = inspect(temp, inspector)
        if not checked['compatible']:
            raise ValueError(checked['error'] + '; metadata cannot repair weights, unsupported tasks or embedded NMS; re-export from the checkpoint')
        restored = onnx.load(temp, load_external_data=False); restored.ClearField('metadata_props')
        if original_body.SerializeToString() != restored.SerializeToString():
            raise ValueError('Non-metadata model fields changed')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        before = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
        after = ort.InferenceSession(str(temp), options, providers=['CPUExecutionProvider'])
        shape = [1, 3, input_size, input_size]
        for axis, dim in enumerate(before.get_inputs()[0].shape):
            if isinstance(dim, int) and dim > 0:
                shape[axis] = dim
        rng = np.random.default_rng(26)
        for data in (np.zeros(shape, dtype=np.float32), np.full(shape, np.float32(114) / 255, dtype=np.float32), rng.random(shape, dtype=np.float32)):
            a = before.run(None, {before.get_inputs()[0].name: data})
            b = after.run(None, {after.get_inputs()[0].name: data})
            if len(a) != len(b) or any(x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)):
                raise ValueError('Prepared model tensor outputs differ from the original')
        # Windows rename fails if the destination appeared meanwhile. No replace.
        if output.exists():
            raise ValueError('Prepared destination appeared during validation')
        if os.name == 'nt':
            temp.rename(output)
        else:
            os.link(temp, output); temp.unlink()
        temp = None
        checked.update(model=str(output), sha256=sha256(output), source=str(source), source_sha256=sha256(source),
                       metadata_only=True, non_metadata_fields_identical=True, tensor_comparisons=3, onnxruntime=ort.__version__)
        return checked
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'prepare'))
    parser.add_argument('model', type=Path)
    parser.add_argument('--inspector', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--format', choices=('dense', 'objectness', 'end2end'))
    parser.add_argument('--layout', choices=('BCN', 'BNC'))
    parser.add_argument('--classes', type=int)
    parser.add_argument('--input-size', type=int, choices=(320, 640), default=640)
    args = parser.parse_args()
    try:
        inspector = inspector_path(args.inspector)
        if args.command == 'prepare':
            if args.output is None:
                parser.error('prepare requires --output')
            report = prepare(args.model, args.output, inspector, args.format, args.classes, args.layout, args.input_size)
        else:
            report = inspect(args.model, inspector)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report['compatible'] else 2
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({'compatible': False, 'error': str(exc)}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    sys.exit(main())
