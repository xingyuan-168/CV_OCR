"""Compare actual DLL detections against a pinned ORT reference and legacy DLL.

Run in the isolated reference environment (ORT1.24.4). Images are converted to
BMP bytes in memory. User inputs are never changed. GPU checks are opt-in by
--devices; the selected provider is recorded, not inferred from the option.
"""
from __future__ import annotations
import argparse
import ctypes as c
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Native:
    def __init__(self, path):
        self.directory = os.add_dll_directory(str(Path(path).resolve().parent))
        self.dll = c.WinDLL(str(Path(path).resolve()))
        for name in ('AI_GetVersion', 'AI_GetLastError'):
            getattr(self.dll, name).restype = c.c_char_p
        self.dll.YOLO_Create.argtypes = [c.POINTER(c.c_int32)]
        self.dll.YOLO_LoadModelFromMemory.argtypes = [c.c_int32,c.c_void_p,c.c_int32,c.c_void_p,c.c_int32,c.c_int32,c.c_int32,c.c_int32,c.c_int32]
        self.dll.YOLO_LoadModelFromPath.argtypes = [c.c_int32,c.c_char_p,c.c_char_p,c.c_int32,c.c_int32,c.c_int32,c.c_int32]
        self.dll.YOLO_InferJson.argtypes = [c.c_int32,c.c_void_p,c.c_int32,c.c_float,c.c_int32,c.c_int32]
        self.dll.YOLO_InferJson.restype = c.c_char_p
        self.dll.YOLO_GetRuntimeStatusJson.argtypes = [c.c_int32,c.c_void_p,c.c_int32]
        self.dll.YOLO_Release.argtypes = [c.c_int32]

    def load(self, path, device, size=0, mode='memory', labels=b'', slots=2):
        handle = c.c_int32()
        assert self.dll.YOLO_Create(c.byref(handle)) == 0
        try:
            if mode == 'memory':
                data = Path(path).read_bytes(); model = c.create_string_buffer(data)
                names = c.create_string_buffer(labels) if labels else None
                status = self.dll.YOLO_LoadModelFromMemory(handle.value,model,len(data),names,len(labels),size,device,0,slots)
            else:
                status = self.dll.YOLO_LoadModelFromPath(handle.value,str(Path(path).resolve()).encode('utf-8'),None,size,device,0,slots)
            if status < 0:
                raise RuntimeError(self.dll.AI_GetLastError().decode('utf-8', errors='replace'))
            return handle.value
        except BaseException:
            self.dll.YOLO_Release(handle.value)
            raise

    def infer(self, handle, bmp, conf=.5, origin=(0,0)):
        buffer = c.create_string_buffer(bmp)
        value = self.dll.YOLO_InferJson(handle, buffer, len(bmp), conf, *origin)
        if not value:
            raise RuntimeError(self.dll.AI_GetLastError().decode('utf-8', errors='replace'))
        return value.decode('mbcs')

    def status(self, handle):
        buffer = c.create_string_buffer(16384)
        assert self.dll.YOLO_GetRuntimeStatusJson(handle,buffer,len(buffer)) == 0
        return json.loads(buffer.value)


def frame(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    ok, encoded = cv2.imencode('.bmp', image); assert ok
    return image, encoded.tobytes()


def preprocess(image, size):
    import numpy as np
    height, width = image.shape[:2]
    scale = np.minimum(np.float32(size)/np.float32(width),np.float32(size)/np.float32(height))
    rw = int(np.floor(np.float32(width)*scale+np.float32(.5)))
    rh = int(np.floor(np.float32(height)*scale+np.float32(.5)))
    px, py = np.float32(size-rw)/2, np.float32(size-rh)/2
    xs = np.minimum((np.arange(rw,dtype=np.float32)/scale).astype(np.int32),width-1)
    ys = np.minimum((np.arange(rh,dtype=np.float32)/scale).astype(np.int32),height-1)
    data = np.full((size,size,3),np.float32(114)/np.float32(255),dtype=np.float32)
    data[int(py):int(py)+rh,int(px):int(px)+rw] = image[ys[:,None],xs[None,:],::-1].astype(np.float32)/np.float32(255)
    return np.ascontiguousarray(data.transpose(2,0,1)[None]), (scale,px,py)


def reference(session, image, descriptor, size, conf=.5):
    import numpy as np
    data, (scale,px,py) = preprocess(image,size)
    result = session.run(None,{session.get_inputs()[0].name:data})[0][0]
    rows = result.T if descriptor['layout']=='BCN' else result
    if descriptor['format']=='end2end':
        scores, ids, boxes = rows[:,4],rows[:,5].astype(np.int32),rows[:,:4].copy()
    else:
        extra = 5 if descriptor['format']=='objectness' else 4
        ids = rows[:,extra:].argmax(axis=1); scores = rows[np.arange(len(rows)),extra+ids].copy()
        if extra==5: scores *= rows[:,4]
        boxes = np.concatenate((rows[:,:2]-rows[:,2:4]*np.float32(.5),rows[:,:2]+rows[:,2:4]*np.float32(.5)),axis=1)
    chosen = scores>=conf
    boxes,scores,ids = boxes[chosen],scores[chosen],ids[chosen]
    boxes[:,[0,2]] = np.clip((boxes[:,[0,2]]-px)/scale,0,image.shape[1])
    boxes[:,[1,3]] = np.clip((boxes[:,[1,3]]-py)/scale,0,image.shape[0])
    order = np.argsort(-scores,kind='stable'); keep=[]
    for i in order:
        if descriptor['format']!='end2end' and keep:
            other=boxes[keep]; wh=np.maximum(np.minimum(boxes[i,2:],other[:,2:])-np.maximum(boxes[i,:2],other[:,:2]),0)
            intersection=wh.prod(axis=1); union=(boxes[i,2:]-boxes[i,:2]).prod()+(other[:,2:]-other[:,:2]).prod(axis=1)-intersection
            iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
            if (iou>.45).any():continue
        keep.append(int(i))
    return [dict(x1=float(boxes[i,0]),y1=float(boxes[i,1]),x2=float(boxes[i,2]),y2=float(boxes[i,3]),score=float(scores[i]),class_id=int(ids[i])) for i in keep]


def compare(actual, expected, coordinate=.01, score=1e-5):
    if len(actual)!=len(expected): raise AssertionError(f'Detection counts differ: {len(actual)} vs {len(expected)}')
    remaining=list(expected)
    for item in actual:
        if set(item)!={'x1','y1','x2','y2','cx','cy','score','class_id','label'}: raise AssertionError('Public JSON fields changed')
        if not 0<=item['score']<=1: raise AssertionError('Invalid score')
        match=next((i for i,e in enumerate(remaining) if item['class_id']==e['class_id'] and abs(item['score']-e['score'])<=score
                    and all(abs(item[k]-e[k])<=coordinate for k in ('x1','y1','x2','y2'))),None)
        if match is None: raise AssertionError('Detection differs from pinned reference: '+str(item))
        remaining.pop(match)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--inspector',type=Path)
    p.add_argument('--devices',default='2,1');p.add_argument('--legacy-only',action='store_true')
    a=p.parse_args();cases=json.loads(a.cases.read_text(encoding='utf-8'))
    os.environ['CQ_AI_YOLO_CPU_THREADS']='1';os.environ['CQ_AI_YOLO_PRECISION']='fp32';os.environ['CQ_AI_YOLO_CUDA_GRAPH']='0'
    native=Native(a.dll);reports=[]
    for case in cases:
        if a.legacy_only and not case.get('legacy'):continue
        descriptor=None;session=None
        if not a.legacy_only:
            import onnxruntime as ort
            assert ort.__version__=='1.24.4', 'Reference must use the pinned delivery ORT version'
            descriptor=json.loads(subprocess.check_output([str(a.inspector.resolve()),case['model']],text=True,encoding='utf-8'))
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            session=ort.InferenceSession(case['model'],options,providers=['CPUExecutionProvider'])
        images=[frame(path) for path in case['images']]
        for device in map(int,a.devices.split(',')):
            records={}
            for mode in ('path','memory'):
                handle=native.load(case['model'],device,case.get('input_size',0),mode)
                try:
                    runtime=native.status(handle)
                    results=[native.infer(handle,bmp) for image,bmp in images]
                    if session:
                        size=runtime['input_width']
                        for (image,bmp),result in zip(images,results):
                            compare(json.loads(result),reference(session,image,descriptor,size),.01 if device==2 else 1,1e-5 if device==2 else .002)
                    # Repeat changing images across five threads sharing the handle.
                    def lane(index):
                        for n in range(15):
                            frame_index=(index+n)%len(images)
                            if native.infer(handle,images[frame_index][1])!=results[frame_index]:raise AssertionError('Concurrent result mismatch')
                    with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lane,range(5)))
                    shifted=json.loads(native.infer(handle,images[0][1],origin=(17,-9)))
                    for x,y in zip(json.loads(results[0]),shifted):
                        for key,delta in [('x1',17),('x2',17),('cx',17),('y1',-9),('y2',-9),('cy',-9)]:
                            if abs(x[key]+delta-y[key])>.02:raise AssertionError('Origin behavior changed')
                    records[mode]=results
                finally:native.dll.YOLO_Release(handle)
            if records['path']!=records['memory']:raise AssertionError('Path and memory results differ')
            reports.append(dict(case=case['name'],device=device,model_sha256=digest(case['model']),descriptor=descriptor,runtime=runtime,results=records['memory'],concurrent_calls=150))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(dict(version=native.dll.AI_GetVersion().decode(),dll_sha256=digest(a.dll),errors=0,cases=reports),ensure_ascii=True,indent=2),encoding='utf-8')
    print(f'Passed {len(reports)} model/backend cases, path+memory and five-thread equality')


if __name__=='__main__':main()
