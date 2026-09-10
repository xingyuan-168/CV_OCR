"""使用 DLL 的纯 CV 路径从图片中提取笔画轨迹。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ai_engine import Engine, image_from_bmp_bytes  # noqa: E402


def main() -> int:
    """解析参数、提取轨迹 JSON，并按需写入磁盘。"""
    parser = argparse.ArgumentParser(description="使用纯 CV 路径提取前景笔画轨迹。")
    parser.add_argument("--image", type=Path, default=ROOT / "examples" / "ocr_test_abc123.bmp")
    parser.add_argument("--threshold", type=int, default=-1, help="0-255，或 -1 表示 Otsu 自动阈值")
    parser.add_argument("--light-on-dark", action="store_true", help="深色背景上的浅色前景")
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None, help="可选 JSON 输出路径")
    args = parser.parse_args()

    os.chdir(ROOT)
    engine = Engine(dll_path=ROOT / "ai_engine.dll", dll_dirs=[ROOT])
    image = image_from_bmp_bytes(args.image.read_bytes())
    result = engine.cv_extract_trace(
        image,
        threshold=args.threshold,
        invert=not args.light_on_dark,
        max_points=args.max_points,
    )

    print("路径数量:", result["path_count"])
    print("点数量:", result["point_count"])
    print("阈值:", result["threshold"])
    if result["paths"]:
        first = result["paths"][0]
        print("第一条路径边界:", first["bounds"])
        print("第一条路径前几个点:", first["points"][:12])

    if args.output is not None:
        args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("已写入:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
