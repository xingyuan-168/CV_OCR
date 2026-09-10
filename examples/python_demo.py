"""最小 Python 冒烟示例：调用 CV、mock YOLO 和 mock OCR。"""

from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ai_engine import AI_DEVICE_CPU, AI_IMAGE_BGR24, Engine, image_from_buffer


def make_test_image(width: int = 64, height: int = 32):
    """创建一张由 Python 持有内存的小型合成 BGR 图。"""
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            pixels[i + 0] = x
            pixels[i + 1] = y
            pixels[i + 2] = 200
    return image_from_buffer(pixels, width, height, width * 3, AI_IMAGE_BGR24)


def main() -> int:
    """使用 mock 配置加载 DLL，并调用代表性的封装方法。"""
    os.chdir(ROOT)
    dll_path = ROOT / "ai_engine.dll"
    config_path = Path("configs") / "mock.ini"

    try:
        engine = Engine(dll_path=dll_path, auto_init=True, config_path=config_path, runtime_device=AI_DEVICE_CPU)
    except OSError as exc:
        print(f"加载失败 {dll_path}: {exc}")
        print("如果该 DLL 链接了 ONNX Runtime，请把 onnxruntime.dll 放到 ai_engine.dll 旁边，或传入 dll_dirs=[...]")
        return 1

    image = make_test_image()
    print("版本:", engine.version())

    stats = engine.cv_mean_color(image, roi=(8, 4, 16, 8))
    print("BGR 均值:", round(stats.mean_b, 2), round(stats.mean_g, 2), round(stats.mean_r, 2))

    model = engine.yolo_model(image.image.width, image.image.height, AI_DEVICE_CPU, 0, 1, 1, 0.45)
    model.load_model_from_memory(b"mock", config_path)
    boxes = model.infer(image, 0.25)
    print("mock yolo 结果:", boxes)

    lines = engine.ocr_recognize(image, min_confidence=0.0)
    print("mock ocr 结果:", lines)

    model.release()
    engine.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
