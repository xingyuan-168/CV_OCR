"""从内存 ONNX 模型运行 PP-OCRv6 识别。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import ai_engine as ai  # noqa: E402


def _default_model(root: Path) -> Path | None:
    """返回仓库内置的 FP32 OCR 识别模型。"""
    path = root / "models" / "ocr_ppocrv6" / "rec.fp32.onnx"
    return path if path.exists() else None


def _load_image(path: Path) -> ai.ImageData:
    """直接加载 BMP；其他格式交给 OpenCV 读取。"""
    if path.suffix.lower() == ".bmp":
        return ai.image_from_bmp_bytes(path.read_bytes())
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("非 BMP 图片需要安装 opencv-python") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return ai.image_from_numpy(image)


def main() -> int:
    """解析命令行参数，从字节加载 OCR 模型，并打印识别输出。"""
    parser = argparse.ArgumentParser(description="PP-OCRv6 纯识别内存加载示例。")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    config = args.config or (root / "configs" / "ocr_ppocrv6_tiny.ini")
    model = args.model or _default_model(root)
    if model is None:
        raise FileNotFoundError(
            "未找到 OCR ONNX 模型。请先将 PP-OCRv6 模型放到 models/ocr_ppocrv6 下，或改用 DLL 内置模型。"
        )

    os.chdir(root)
    image = _load_image(args.image)
    model_bytes = model.read_bytes()

    with ai.Engine(dll_path=root / "ai_engine.dll", dll_dirs=[root]) as engine:
        engine.ocr_load_models_from_memory(None, model_bytes, config, ai.AI_DEVICE_AUTO)
        text = engine.ocr_recognize_line(
            image, min_confidence=0.0, output_format=ai.AI_OCR_OUTPUT_TEXT
        )
        lines = engine.ocr_recognize(image, min_confidence=0.0)
        latency_us = engine.last_latency_us(ai.AI_MODULE_OCR)

    print("图片:", args.image)
    print("模型:", model)
    print("耗时_微秒:", latency_us)
    print("文本:", text)
    print("文本行:", lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
