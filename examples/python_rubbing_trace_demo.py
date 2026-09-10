"""将源字符 ROI 的笔画轨迹映射到目标 ROI。"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ai_engine import AI_IMAGE_GRAY8, Engine, image_from_buffer  # noqa: E402


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """解析命令行传入的 x,y,w,h ROI，并校验宽高为正数。"""
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI 宽度和高度必须为正数")
    return x, y, w, h


def load_bgr_image(path: Path) -> tuple[bytearray, int, int]:
    """将图片加载为自上而下紧凑 BGR 像素，优先使用无依赖 BMP 路径。"""
    data = path.read_bytes()
    if len(data) >= 54 and data[:2] == b"BM":
        return load_bgr_bmp(data)

    try:
        from PIL import Image  # type: ignore

        rgb = Image.open(path).convert("RGB")
        width, height = rgb.size
        raw = rgb.tobytes()
        bgr = bytearray(width * height * 3)
        for i in range(width * height):
            bgr[i * 3 + 0] = raw[i * 3 + 2]
            bgr[i * 3 + 1] = raw[i * 3 + 1]
            bgr[i * 3 + 2] = raw[i * 3 + 0]
        return bgr, width, height
    except ImportError:
        pass

    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        height, width = image.shape[:2]
        return bytearray(image.tobytes()), width, height
    except ImportError as exc:
        raise RuntimeError("非 BMP 图片需要 Pillow 或 opencv-python；如需避免依赖，请将截图保存为 BMP") from exc


def load_bgr_bmp(data: bytes) -> tuple[bytearray, int, int]:
    """将未压缩 24/32 位 BMP 字节解码为自上而下 BGR 像素。"""
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40:
        raise ValueError("不支持的 BMP DIB 头")

    width = struct.unpack_from("<i", data, 18)[0]
    raw_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if width <= 0 or raw_height == 0 or planes != 1 or compression != 0:
        raise ValueError("不支持的 BMP 布局")
    if bpp not in (24, 32):
        raise ValueError("仅支持 24 位和 32 位 BMP 图片")

    height = abs(raw_height)
    channels = bpp // 8
    stride = ((width * channels + 3) // 4) * 4
    out = bytearray(width * height * 3)
    for y in range(height):
        src_y = height - 1 - y if raw_height > 0 else y
        src = pixel_offset + src_y * stride
        for x in range(width):
            si = src + x * channels
            di = (y * width + x) * 3
            out[di : di + 3] = data[si : si + 3]
    return out, width, height


def write_mask_bmp(path: Path, mask: bytearray, width: int, height: int) -> None:
    """将灰度掩码写成简单 24 位 BMP，方便目视调试。"""
    stride = ((width * 3 + 3) // 4) * 4
    pixel_size = stride * height
    file_size = 14 + 40 + pixel_size
    header = bytearray()
    header += b"BM"
    header += struct.pack("<IHHI", file_size, 0, 0, 54)
    header += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_size, 0, 0, 0, 0)
    pixels = bytearray(pixel_size)
    for y in range(height):
        dst_y = height - 1 - y
        row = dst_y * stride
        for x in range(width):
            value = mask[y * width + x]
            i = row + x * 3
            pixels[i + 0] = value
            pixels[i + 1] = value
            pixels[i + 2] = value
    path.write_bytes(header + pixels)


def bgr_to_gray(b: int, g: int, r: int) -> int:
    """使用与 DLL 相同的整数权重，将一个 BGR 像素转为亮度。"""
    return (29 * b + 150 * g + 77 * r) >> 8


def make_foreground_mask(
    bgr: bytearray,
    image_width: int,
    image_height: int,
    roi: tuple[int, int, int, int],
    mode: str,
    green_min: int,
    green_delta: int,
    gray_threshold: int,
) -> bytearray:
    """为选定源字符 ROI 构建二值前景掩码。"""
    x0, y0, w, h = roi
    if x0 < 0 or y0 < 0 or x0 + w > image_width or y0 + h > image_height:
        raise ValueError("ROI 超出源图范围")

    mask = bytearray([255]) * (w * h)
    for y in range(h):
        src_y = y0 + y
        for x in range(w):
            src_x = x0 + x
            si = (src_y * image_width + src_x) * 3
            b = bgr[si + 0]
            g = bgr[si + 1]
            r = bgr[si + 2]
            if mode == "green":
                is_foreground = g >= green_min and g >= r + green_delta and g >= b + green_delta
            elif mode == "dark":
                is_foreground = bgr_to_gray(b, g, r) <= gray_threshold
            elif mode == "bright":
                is_foreground = bgr_to_gray(b, g, r) >= gray_threshold
            else:
                raise ValueError(f"不支持的模式: {mode}")
            if is_foreground:
                mask[y * w + x] = 0
    return mask


def map_point(point: list[int], left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> list[int]:
    """将点从左侧 ROI 坐标按比例映射到源图上的右侧 ROI。"""
    _, _, left_w, left_h = left
    right_x, right_y, right_w, right_h = right
    x, y = point
    mapped_x = right_x + round(x * (right_w - 1) / max(1, left_w - 1))
    mapped_y = right_y + round(y * (right_h - 1) / max(1, left_h - 1))
    return [mapped_x, mapped_y]


def bounds_for(points: list[list[int]]) -> dict[str, int]:
    """计算包含点集的最小 x/y/w/h 边界。"""
    if not points:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs) + 1, "h": max(ys) - min(ys) + 1}


def main() -> int:
    """提取源笔画、映射到目标 ROI，并写出 JSON。"""
    parser = argparse.ArgumentParser(description="提取左侧拓印字符，并将笔画映射到右侧目标框。")
    parser.add_argument("--image", type=Path, required=True, help="完整截图图片；BMP 无需额外依赖")
    parser.add_argument("--left", type=parse_roi, required=True, help="左侧字符 ROI：x,y,w,h")
    parser.add_argument("--right", type=parse_roi, required=True, help="右侧目标 ROI：x,y,w,h")
    parser.add_argument("--mode", choices=("green", "dark", "bright"), default="green")
    parser.add_argument("--green-min", type=int, default=120)
    parser.add_argument("--green-delta", type=int, default=25)
    parser.add_argument("--gray-threshold", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("rubbing_trace.json"))
    parser.add_argument("--debug-mask", type=Path, default=None, help="可选 BMP 掩码输出，用于检查提取质量")
    args = parser.parse_args()

    os.chdir(ROOT)
    bgr, width, height = load_bgr_image(args.image)
    mask = make_foreground_mask(
        bgr,
        width,
        height,
        args.left,
        args.mode,
        args.green_min,
        args.green_delta,
        args.gray_threshold,
    )
    if args.debug_mask is not None:
        write_mask_bmp(args.debug_mask, mask, args.left[2], args.left[3])

    mask_image = image_from_buffer(mask, args.left[2], args.left[3], args.left[2], AI_IMAGE_GRAY8)
    engine = Engine(dll_path=ROOT / "ai_engine.dll", dll_dirs=[ROOT])
    trace = engine.cv_extract_trace(mask_image, threshold=128, invert=True, max_points=args.max_points)

    mapped_paths = []
    for path in trace["paths"]:
        points = [map_point(point, args.left, args.right) for point in path["points"]]
        if not points:
            continue
        mapped_paths.append({"bounds": bounds_for(points), "points": points})

    result = {
        "image": str(args.image),
        "image_size": {"w": width, "h": height},
        "left_roi": {"x": args.left[0], "y": args.left[1], "w": args.left[2], "h": args.left[3]},
        "right_roi": {"x": args.right[0], "y": args.right[1], "w": args.right[2], "h": args.right[3]},
        "mode": args.mode,
        "source_trace": trace,
        "mapped_paths": mapped_paths,
        "path_count": len(mapped_paths),
        "point_count": sum(len(path["points"]) for path in mapped_paths),
    }

    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("图片尺寸:", width, height)
    print("路径数量:", result["path_count"])
    print("点数量:", result["point_count"])
    print("输出:", args.output)
    if args.debug_mask is not None:
        print("调试掩码:", args.debug_mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
