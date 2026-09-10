"""通过 Python 封装运行 SxR YOLO 模型。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ai_engine import AI_DEVICE_AUTO, Engine, image_from_bmp_bytes


def find_sample_image() -> Path:
    """优先查找打包示例 BMP；不存在时回退到本地训练图片目录。"""
    packaged_sample = ROOT / "examples" / "sample_sxr.bmp"
    if packaged_sample.exists():
        return packaged_sample

    dataset_dir = ROOT / "models" / "models_SXR_带标注" / "images"
    for path in sorted(dataset_dir.glob("*.bmp")):
        return path
    raise FileNotFoundError(dataset_dir)


def main() -> int:
    """加载示例图片，执行 YOLO 推理，并打印检测框和耗时。"""
    os.chdir(ROOT)
    image_path = find_sample_image()
    image = image_from_bmp_bytes(image_path.read_bytes())

    with Engine(dll_path=ROOT / "ai_engine.dll") as engine:
        model = engine.yolo_model(0, 0, AI_DEVICE_AUTO, 0, 1, 0, 0.45)
        model.load_model(ROOT / "models" / "yolo" / "best.onnx", ROOT / "configs" / "yolo_sxr.ini")
        boxes = model.infer(image, 0.25)
        latency_us = model.last_latency_us()
        model.release()

    print("图片:", image_path)
    print("尺寸:", image.image.width, image.image.height)
    print("耗时_微秒:", latency_us)
    print("检测框数量:", len(boxes))
    for box in boxes:
        print(box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
