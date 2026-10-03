"""CQ_AI 公开 C ABI 的 Python ctypes 封装。

本模块映射稳定的 C ABI，并补充 BMP、NumPy、内存图像、结果转换、
DLL 依赖目录自动搜索等 Python 侧辅助能力。
"""

from __future__ import annotations

import ctypes as C
import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


AIENGINE_VERSION_MAJOR = 0
AIENGINE_VERSION_MINOR = 14
AIENGINE_VERSION_PATCH = 5
AIENGINE_VERSION = (AIENGINE_VERSION_MAJOR, AIENGINE_VERSION_MINOR, AIENGINE_VERSION_PATCH)

AI_OK = 0
AI_ERR_INVALID_ARGUMENT = -1
AI_ERR_NOT_INITIALIZED = -2
AI_ERR_BACKEND_NOT_CONFIGURED = -3
AI_ERR_IMAGE_FORMAT = -4
AI_ERR_BUFFER_TOO_SMALL = -5
AI_ERR_CONFIG = -6
AI_ERR_INVALID_HANDLE = -7
AI_ERR_ALREADY_LOADED = -8
AI_ERR_BUSY = -9
AI_ERR_RUNTIME = -100

AI_IMAGE_GRAY8 = 1
AI_IMAGE_BGR24 = 2
AI_IMAGE_BGRA32 = 3
AI_IMAGE_RGB24 = 4
AI_IMAGE_RGBA32 = 5

AI_MODULE_CV = 1
AI_MODULE_OCR = 2
AI_MODULE_YOLO = 3
AI_OCR_STAGE_DETECTION = 1
AI_OCR_STAGE_RECOGNITION = 2
AI_OCR_STAGE_POSTPROCESS = 3

AI_DEVICE_AUTO = 0
AI_DEVICE_DIRECTML = 1
AI_DEVICE_CPU = 2
AI_DEVICE_TENSORRT = 3

AI_OCR_OUTPUT_TEXT = 1
AI_OCR_OUTPUT_JSON = 2

AIENGINE_MAX_LABEL = 32
AIENGINE_MAX_TEXT = 512


class AIEngineError(RuntimeError):
    """DLL 调用返回负数 ai_engine 状态码时抛出的异常。"""

    def __init__(self, status: int, message: str):
        """构造异常，同时保留原始状态码和最近错误文本。"""
        self.status = status
        super().__init__(f"ai_engine status {status}: {message}")


class AIImage(C.Structure):
    """C ABI 图像视图；像素内存由调用方持有，必须在 DLL 调用期间保持有效。"""

    _pack_ = 1
    _fields_ = [
        ("data", C.POINTER(C.c_uint8)),
        ("width", C.c_int32),
        ("height", C.c_int32),
        ("stride", C.c_int32),
        ("format", C.c_int32),
    ]


class AIRect(C.Structure):
    """C ABI 矩形，包含整数 x/y 起点和宽高。"""

    _pack_ = 1
    _fields_ = [
        ("x", C.c_int32),
        ("y", C.c_int32),
        ("w", C.c_int32),
        ("h", C.c_int32),
    ]


class AIDetectBox(C.Structure):
    """C ABI YOLO 检测结果框。"""

    _pack_ = 1
    _fields_ = [
        ("x1", C.c_float),
        ("y1", C.c_float),
        ("x2", C.c_float),
        ("y2", C.c_float),
        ("score", C.c_float),
        ("class_id", C.c_int32),
        ("label", C.c_char * AIENGINE_MAX_LABEL),
    ]


class AIOcrLine(C.Structure):
    """C ABI OCR 文本行结果，包含边界框、置信度和 UTF-8 文本。"""

    _pack_ = 1
    _fields_ = [
        ("box", AIRect),
        ("confidence", C.c_float),
        ("text", C.c_char * AIENGINE_MAX_TEXT),
    ]


class AIOcrRuntimeOptions(C.Structure):
    """OCR 内置模型性能和检测后处理参数；0 表示自动或模型默认。"""

    _pack_ = 1
    _fields_ = [
        ("det_input_width", C.c_int32),
        ("det_input_height", C.c_int32),
        ("intra_op_threads", C.c_int32),
        ("det_binary_threshold", C.c_float),
        ("det_box_score_threshold", C.c_float),
        ("det_unclip_ratio", C.c_float),
    ]


class AIColorStats(C.Structure):
    """C ABI 颜色统计结果，用于整图或 ROI。"""

    _pack_ = 1
    _fields_ = [
        ("mean_b", C.c_double),
        ("mean_g", C.c_double),
        ("mean_r", C.c_double),
        ("min_gray", C.c_double),
        ("max_gray", C.c_double),
        ("pixels", C.c_int32),
    ]


class AIColorFindResult(C.Structure):
    """C ABI 找色结果，包含数量和边界矩形。"""

    _pack_ = 1
    _fields_ = [
        ("count", C.c_int32),
        ("ratio", C.c_float),
        ("first_x", C.c_int32),
        ("first_y", C.c_int32),
        ("bounds", AIRect),
    ]


class AIImageMatch(C.Structure):
    """C ABI 模板匹配结果。"""

    _pack_ = 1
    _fields_ = [
        ("box", AIRect),
        ("score", C.c_float),
        ("template_index", C.c_int32),
    ]


class CVMatchResult(C.Structure):
    """易语言 CV 模板查找结果。"""

    _pack_ = 1
    _fields_ = [
        ("x", C.c_int32),
        ("y", C.c_int32),
        ("w", C.c_int32),
        ("h", C.c_int32),
        ("sim", C.c_float),
        ("template_index", C.c_int32),
    ]


class OCRTextResult(C.Structure):
    """易语言 OCR 单文本结果；坐标、尺寸和置信度与 C ABI 完全一致。"""

    _pack_ = 1
    _fields_ = [
        ("cx", C.c_int32),
        ("cy", C.c_int32),
        ("x", C.c_int32),
        ("y", C.c_int32),
        ("w", C.c_int32),
        ("h", C.c_int32),
        ("score", C.c_float),
    ]


class OCRCoordResult(C.Structure):
    """易语言 OCR 坐标结果；target_index 是输入目标的零基序号。"""

    _pack_ = 1
    _fields_ = [
        ("x", C.c_int32),
        ("y", C.c_int32),
        ("w", C.c_int32),
        ("h", C.c_int32),
        ("target_index", C.c_int32),
    ]


@dataclass
class ImageData:
    """Python 持有的图像视图，以及用于保持像素内存生命周期的对象。"""

    image: AIImage
    keepalive: object


def _project_root() -> Path:
    """根据当前封装文件推断项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _bundled_native_dir() -> Path:
    """返回离线 Wheel 中原生 DLL 的固定目录。"""
    return Path(__file__).resolve().parent / "cq_ai_engine" / "_native"


PathLikeValue = Union[os.PathLike, str, bytes]
TextValue = Union[str, bytes]


def _bytes_path(value: Optional[PathLikeValue]) -> Optional[bytes]:
    """为标准 AI_* C ABI 编码 UTF-8 文件系统路径。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return os.fspath(value).encode("utf-8")


def _utf8(value: Optional[TextValue]) -> Optional[bytes]:
    """将可选文本编码为 C ABI 使用的 UTF-8 字节。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def _compat_text(value: Optional[Union[os.PathLike, TextValue]]) -> Optional[bytes]:
    """编码易语言兼容入口文本；Windows 使用当前 ANSI 代码页。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return os.fspath(value).encode("mbcs" if os.name == "nt" else "utf-8")


def _rgb_text(value: Union[int, TextValue]) -> bytes:
    """把整数或文本透明色规范为 DLL 接受的 RGB 十六进制文本。"""
    if isinstance(value, int):
        if value < 0 or value > 0xFFFFFF:
            raise ValueError("transparent_rgb must be within 0x000000-0xFFFFFF")
        return f"{value:06X}".encode("ascii")
    encoded = _utf8(value)
    if not encoded:
        raise ValueError("transparent_rgb cannot be empty")
    return encoded


def _parse_compact_points(text: str, x_key: str, y_key: str) -> List[Dict[str, int]]:
    """解析 v23.5 的 ID,x,y|ID,x,y 紧凑结果文本。"""
    if text == "":
        return []
    results: List[Dict[str, int]] = []
    for item in text.split("|"):
        fields = item.split(",")
        if len(fields) != 3 or any(field == "" for field in fields):
            raise AIEngineError(AI_ERR_RUNTIME, f"malformed compact result: {text!r}")
        try:
            target_id, x, y = (int(field, 10) for field in fields)
        except ValueError as error:
            raise AIEngineError(AI_ERR_RUNTIME, f"malformed compact result: {text!r}") from error
        if target_id < 0:
            raise AIEngineError(AI_ERR_RUNTIME, f"malformed compact result id: {target_id}")
        results.append({"id": target_id, x_key: x, y_key: y})
    return results


def _decode(raw: bytes) -> str:
    """解码 C ABI 中固定大小、以空字符结尾的 UTF-8 字段。"""
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _rect_dict(rect: AIRect) -> Dict[str, int]:
    """将 AIRect 结构体转换为普通 Python 字典。"""
    return {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h}


def _match_dict(match: AIImageMatch) -> Dict[str, object]:
    """将 AIImageMatch 结构体转换为便于 JSON 序列化的字典。"""
    return {
        "box": _rect_dict(match.box),
        "score": float(match.score),
        "template_index": int(match.template_index),
    }


def _as_image(value) -> AIImage:
    """接受 ImageData 或 AIImage，并返回 C ABI 图像视图。"""
    if isinstance(value, ImageData):
        return value.image
    if isinstance(value, AIImage):
        return value
    raise TypeError("需要传入 AIImage 或 ImageData")


def _as_roi(roi: Optional[Union[AIRect, Sequence[int]]]) -> Optional[AIRect]:
    """将可选 ROI 输入归一化为 AIRect 结构体。"""
    if roi is None:
        return None
    if isinstance(roi, AIRect):
        return roi
    if len(roi) != 4:
        raise ValueError("roi 必须是 (x, y, w, h)")
    return AIRect(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))


def image_from_buffer(
    data,
    width: int,
    height: int,
    stride: int,
    image_format: int,
    offset: int = 0,
) -> ImageData:
    """在 Python 字节类像素缓冲区上创建 AIImage 视图。"""
    view = memoryview(data)
    if view.readonly:
        cbuf = (C.c_uint8 * view.nbytes).from_buffer_copy(view)
    else:
        cbuf = (C.c_uint8 * view.nbytes).from_buffer(data)
    ptr = C.cast(C.byref(cbuf, offset), C.POINTER(C.c_uint8))
    image = AIImage(ptr, int(width), int(height), int(stride), int(image_format))
    return ImageData(image, cbuf)


def image_from_bmp_bytes(data) -> ImageData:
    """直接基于未压缩 24/32 位 BMP 字节创建 AIImage 视图。"""
    view = memoryview(data)
    if len(view) < 54 or view[:2].tobytes() != b"BM":
        raise ValueError("需要传入 BMP 字节")

    pixel_offset = struct.unpack_from("<I", view, 10)[0]
    dib_size = struct.unpack_from("<I", view, 14)[0]
    if dib_size < 40:
        raise ValueError("不支持的 BMP DIB 头")

    width = struct.unpack_from("<i", view, 18)[0]
    raw_height = struct.unpack_from("<i", view, 22)[0]
    planes = struct.unpack_from("<H", view, 26)[0]
    bpp = struct.unpack_from("<H", view, 28)[0]
    compression = struct.unpack_from("<I", view, 30)[0]
    if width <= 0 or raw_height == 0 or planes != 1 or compression != 0:
        raise ValueError("不支持的 BMP 布局")

    if bpp == 24:
        channels = 3
        image_format = AI_IMAGE_BGR24
    elif bpp == 32:
        channels = 4
        image_format = AI_IMAGE_BGRA32
    else:
        raise ValueError("仅支持 24 位和 32 位 BMP 图片")

    height = abs(raw_height)
    stride = ((width * channels + 3) // 4) * 4
    if raw_height > 0:
        offset = pixel_offset + (height - 1) * stride
        stride = -stride
    else:
        offset = pixel_offset
    return image_from_buffer(data, width, height, stride, image_format, offset)


def image_from_numpy(array, image_format: Optional[int] = None, copy: bool = True) -> ImageData:
    """从 uint8 NumPy 数组创建 AIImage，支持灰度、BGR 或 BGRA 布局。"""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("image_from_numpy 需要安装 numpy") from exc

    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        raise ValueError("NumPy 图像 dtype 必须是 uint8")

    if arr.ndim == 2:
        height, width = arr.shape
        inferred_format = AI_IMAGE_GRAY8
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        height, width, channels = arr.shape
        if channels == 3:
            inferred_format = AI_IMAGE_BGR24
        else:
            inferred_format = AI_IMAGE_BGRA32
    else:
        raise ValueError("NumPy 图像必须是 HxW、HxWx3 或 HxWx4")

    image_format = inferred_format if image_format is None else int(image_format)
    if copy or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    stride = int(arr.strides[0])
    ptr = arr.ctypes.data_as(C.POINTER(C.c_uint8))
    image = AIImage(ptr, int(width), int(height), stride, image_format)
    return ImageData(image, arr)


def _bmp_buffer(data):
    """为句柄式 CV 接口创建完整 BMP 的临时 C 缓冲区。"""
    view = memoryview(data)
    if view.nbytes <= 0:
        raise ValueError("BMP 缓冲区不能为空")
    buf = (C.c_uint8 * view.nbytes).from_buffer_copy(view)
    return buf, C.cast(buf, C.POINTER(C.c_uint8)), view.nbytes


class Engine:
    """CQ_AI 公开 C ABI 的高级 Python 封装。

    一个 Engine 实例持有已加载的 DLL 句柄，并提供 CV、OCR、YOLO 方法，
    同时处理 Python 风格参数和结果转换。
    """

    def __init__(
        self,
        dll_path: Optional[PathLikeValue] = None,
        dll_dirs: Iterable[PathLikeValue] = (),
        auto_init: bool = False,
        config_path: Optional[PathLikeValue] = None,
        runtime_device: int = AI_DEVICE_AUTO,
    ):
        """加载 DLL，注册依赖目录，并按需自动初始化。"""
        self._dll_dir_handles = []
        self.dll_path = self._resolve_dll_path(dll_path)
        self._add_dll_dir(self.dll_path.parent)
        for dll_dir in dll_dirs:
            self._add_dll_dir(Path(dll_dir))

        loader = C.WinDLL if os.name == "nt" else C.CDLL
        self._dll = loader(str(self.dll_path))
        self._bind_functions()
        if auto_init:
            self.init(config_path, runtime_device)

    def _resolve_dll_path(self, dll_path: Optional[PathLikeValue]) -> Path:
        """从显式路径、Wheel 原生目录或开发目录查找业务 DLL。"""
        if dll_path is not None:
            path = Path(dll_path)
            if not path.exists():
                raise FileNotFoundError(path)
            return path.resolve()

        candidates = [
            _bundled_native_dir() / "CQ_AI_x64.dll",
            Path.cwd() / "CQ_AI_x64.dll",
            _project_root() / "CQ_AI_x64.dll",
            Path(__file__).resolve().parent / "CQ_AI_x64.dll",
            Path.cwd() / "CQ_X86.dll",
            _project_root() / "CQ_X86.dll",
            Path(__file__).resolve().parent / "CQ_X86.dll",
            Path.cwd() / "ai_engine.dll",
            _project_root() / "ai_engine.dll",
            Path(__file__).resolve().parent / "ai_engine.dll",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError("找不到 Wheel 内置 CQ_AI_x64.dll、CQ_X86.dll 或兼容的 ai_engine.dll")

    def _add_dll_dir(self, path: Path) -> None:
        """注册 Windows DLL 依赖目录，并保持目录句柄存活。"""
        if os.name == "nt" and hasattr(os, "add_dll_directory") and path.exists():
            self._dll_dir_handles.append(os.add_dll_directory(str(path)))

    def _bind_functions(self) -> None:
        """声明所有导出 DLL 函数的 ctypes 参数类型和返回类型。"""
        dll = self._dll
        dll.AI_GetVersion.argtypes = []
        dll.AI_GetVersion.restype = C.c_char_p

        dll.AI_Init.argtypes = [C.c_char_p]
        dll.AI_Init.restype = C.c_int32
        dll.AI_InitEx.argtypes = [C.c_char_p, C.c_int32]
        dll.AI_InitEx.restype = C.c_int32
        dll.AI_Release.argtypes = []
        dll.AI_Release.restype = None
        dll.AI_ShutdownWorker.argtypes = []
        dll.AI_ShutdownWorker.restype = C.c_int32
        dll.AI_GetLastError.argtypes = []
        dll.AI_GetLastError.restype = C.c_char_p
        dll.AI_GetLastLatencyUs.argtypes = [C.c_int32]
        dll.AI_GetLastLatencyUs.restype = C.c_int64
        dll.AI_GetOcrStageLatencyUs.argtypes = [C.c_int32]
        dll.AI_GetOcrStageLatencyUs.restype = C.c_int64
        dll.AI_HasEmbeddedAssets.argtypes = []
        dll.AI_HasEmbeddedAssets.restype = C.c_int32
        dll.AI_YoloCreate.argtypes = [C.POINTER(C.c_int32)]
        dll.AI_YoloCreate.restype = C.c_int32
        dll.AI_YoloLoadModel.argtypes = [C.c_int32, C.c_char_p, C.c_char_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
        dll.AI_YoloLoadModel.restype = C.c_int32
        dll.AI_YoloLoadModelFromMemory.argtypes = [C.c_int32, C.c_void_p, C.c_int32, C.c_char_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
        dll.AI_YoloLoadModelFromMemory.restype = C.c_int32
        dll.AI_YoloLoadEmbeddedModel.argtypes = [C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
        dll.AI_YoloLoadEmbeddedModel.restype = C.c_int32
        dll.AI_YoloDetect.argtypes = [C.c_int32, C.POINTER(AIImage), C.c_float]
        dll.AI_YoloDetect.restype = C.c_char_p
        dll.AI_YoloInfer.argtypes = [C.c_int32, C.POINTER(AIImage), C.c_float]
        dll.AI_YoloInfer.restype = C.c_char_p
        dll.AI_YoloRelease.argtypes = [C.c_int32]
        dll.AI_YoloRelease.restype = C.c_int32
        dll.YOLO_GetRuntimeStatusJson.argtypes = [C.c_int32, C.POINTER(C.c_char), C.c_int32]
        dll.YOLO_GetRuntimeStatusJson.restype = C.c_int32
        dll.YOLO_GetLastLatencyUs.argtypes = [C.c_int32]
        dll.YOLO_GetLastLatencyUs.restype = C.c_int64
        dll.AI_OcrLoadModels.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_int32]
        dll.AI_OcrLoadModels.restype = C.c_int32
        dll.AI_OcrLoadModelsFromMemory.argtypes = [
            C.c_void_p,
            C.c_int32,
            C.c_void_p,
            C.c_int32,
            C.c_char_p,
            C.c_int32,
        ]
        dll.AI_OcrLoadModelsFromMemory.restype = C.c_int32
        dll.AI_OcrLoadEmbeddedModels.argtypes = [C.c_int32]
        dll.AI_OcrLoadEmbeddedModels.restype = C.c_int32
        dll.AI_OcrRecognize.argtypes = [C.POINTER(AIImage), C.c_float]
        dll.AI_OcrRecognize.restype = C.c_char_p
        dll.AI_OcrRecognizeLine.argtypes = [C.POINTER(AIImage), C.c_int32, C.c_float, C.POINTER(C.c_char), C.c_int32]
        dll.AI_OcrRecognizeLine.restype = C.c_int32
        dll.AI_OcrRecognizeLines.argtypes = [C.POINTER(AIImage), C.c_int32, C.c_float, C.POINTER(C.c_char), C.c_int32]
        dll.AI_OcrRecognizeLines.restype = C.c_int32
        dll.AI_OcrFindText.argtypes = [C.POINTER(AIImage), C.c_char_p, C.c_float]
        dll.AI_OcrFindText.restype = C.c_char_p
        dll.AI_OcrRelease.argtypes = []
        dll.AI_OcrRelease.restype = C.c_int32
        dll.OCR_LoadEmbeddedModel.argtypes = [C.c_int32, C.c_int32]
        dll.OCR_LoadEmbeddedModel.restype = C.c_int32
        dll.OCR_LoadEmbeddedModelEx.argtypes = [C.c_int32, C.c_int32, C.POINTER(AIOcrRuntimeOptions)]
        dll.OCR_LoadEmbeddedModelEx.restype = C.c_int32
        dll.OCR_Recognize.argtypes = [C.POINTER(C.c_uint8), C.c_int32, C.c_int32, C.c_float, C.c_char_p, C.c_int32, C.c_int32]
        dll.OCR_Recognize.restype = C.c_char_p
        dll.OCR_FindOneText.argtypes = [C.POINTER(C.c_uint8), C.c_int32, C.c_char_p, C.c_float, C.POINTER(OCRTextResult), C.c_char_p, C.c_int32, C.c_int32]
        dll.OCR_FindOneText.restype = C.c_int32
        dll.OCR_FindMultiText.argtypes = [C.POINTER(C.c_uint8), C.c_int32, C.c_char_p, C.c_float, C.c_char_p, C.c_int32, C.c_int32]
        dll.OCR_FindMultiText.restype = C.c_char_p
        dll.OCR_FindOneCoord.argtypes = [C.POINTER(C.c_uint8), C.c_int32, C.c_char_p, C.c_float, C.POINTER(OCRCoordResult), C.c_char_p, C.c_int32, C.c_int32]
        dll.OCR_FindOneCoord.restype = C.c_int32
        dll.YOLO_InferJson.argtypes = [C.c_int32, C.POINTER(C.c_uint8), C.c_int32, C.c_float, C.c_int32, C.c_int32]
        dll.YOLO_InferJson.restype = C.c_char_p

        dll.AI_CvInit.argtypes = []
        dll.AI_CvInit.restype = C.c_int32
        dll.AI_CvToGray.argtypes = [C.POINTER(AIImage), C.POINTER(AIRect), C.POINTER(C.c_uint8), C.c_int32]
        dll.AI_CvToGray.restype = C.c_int32
        dll.AI_CvThreshold.argtypes = [C.POINTER(AIImage), C.POINTER(AIRect), C.c_int32, C.POINTER(C.c_uint8), C.c_int32]
        dll.AI_CvThreshold.restype = C.c_int32
        dll.AI_CvExtractTraceJson.argtypes = [
            C.POINTER(AIImage),
            C.POINTER(AIRect),
            C.c_int32,
            C.c_int32,
            C.c_int32,
            C.POINTER(C.c_char),
            C.c_int32,
        ]
        dll.AI_CvExtractTraceJson.restype = C.c_int32
        dll.AI_CvMeanColor.argtypes = [C.POINTER(AIImage), C.POINTER(AIRect), C.POINTER(AIColorStats)]
        dll.AI_CvMeanColor.restype = C.c_int32
        dll.AI_CvFindColor.argtypes = [C.POINTER(AIImage), C.POINTER(AIRect), C.c_uint32, C.c_int32, C.POINTER(AIColorFindResult)]
        dll.AI_CvFindColor.restype = C.c_int32
        dll.AI_CvFindImage.argtypes = [C.POINTER(AIImage), C.POINTER(AIImage), C.c_float, C.POINTER(AIImageMatch)]
        dll.AI_CvFindImage.restype = C.c_int32
        dll.AI_CvFindImages.argtypes = [C.POINTER(AIImage), C.POINTER(AIImage), C.c_int32, C.c_float]
        dll.AI_CvFindImages.restype = C.c_char_p
        dll.AI_CvFindTransparentImage.argtypes = [
            C.POINTER(AIImage),
            C.POINTER(AIImage),
            C.c_int32,
            C.c_float,
            C.POINTER(AIImageMatch),
        ]
        dll.AI_CvFindTransparentImage.restype = C.c_int32
        dll.AI_CvFindTransparentImages.argtypes = [
            C.POINTER(AIImage), C.POINTER(AIImage), C.c_int32, C.c_int32, C.c_float
        ]
        dll.AI_CvFindTransparentImages.restype = C.c_char_p

        dll.CV_Create.argtypes = [C.POINTER(C.c_int32)]
        dll.CV_Create.restype = C.c_int32
        dll.CV_LoadTemplateDir.argtypes = [C.c_int32, C.c_char_p, C.c_int32]
        dll.CV_LoadTemplateDir.restype = C.c_int32
        dll.CV_LoadTemplateZipFromMemory.argtypes = [C.c_int32, C.POINTER(C.c_uint8), C.c_int32]
        dll.CV_LoadTemplateZipFromMemory.restype = C.c_int32
        dll.CV_ClearTemplateCache.argtypes = [C.c_int32]
        dll.CV_ClearTemplateCache.restype = C.c_int32
        dll.CV_Release.argtypes = [C.c_int32]
        dll.CV_Release.restype = C.c_int32
        dll.CV_FindOne.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_uint8), C.c_int32, C.c_float, C.c_int32, C.POINTER(CVMatchResult), C.c_int32, C.c_int32]
        dll.CV_FindOne.restype = C.c_int32
        dll.CV_FindTransparentOne.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_uint8), C.c_int32, C.c_float, C.c_int32, C.c_char_p, C.POINTER(CVMatchResult), C.c_int32, C.c_int32]
        dll.CV_FindTransparentOne.restype = C.c_int32
        dll.CV_FindMultiText.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_uint8), C.c_int32, C.c_char_p, C.c_float, C.c_int32, C.c_int32, C.c_int32]
        dll.CV_FindMultiText.restype = C.c_char_p
        dll.CV_FindTransparentMultiText.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_uint8), C.c_int32, C.c_char_p, C.c_float, C.c_char_p, C.c_int32, C.c_int32]
        dll.CV_FindTransparentMultiText.restype = C.c_char_p

    def _check(self, status: int) -> int:
        """状态码为负数时抛出 AIEngineError；成功值原样返回。"""
        if status < 0:
            raise AIEngineError(status, self.last_error())
        return status

    def version(self) -> str:
        """返回 ai_engine DLL 版本字符串。"""
        return self._dll.AI_GetVersion().decode("utf-8", errors="replace")

    def init(self, config_path: Optional[PathLikeValue] = None, runtime_device: int = AI_DEVICE_AUTO) -> None:
        """使用可选 INI 配置和设备模式初始化共享 DLL 引擎。"""
        self._check(self._dll.AI_InitEx(_bytes_path(config_path), int(runtime_device)))

    def release(self) -> None:
        """释放共享 DLL 引擎持有的全部模块后端。"""
        self._dll.AI_Release()

    def shutdown_worker(self) -> None:
        """兼容旧代理 ABI；x64 直连 DLL 中固定为空操作。"""
        self._check(self._dll.AI_ShutdownWorker())

    def close(self) -> None:
        """兼容上下文管理器的 release() 别名。"""
        self.release()

    def __enter__(self) -> "Engine":
        """进入 with 代码块时返回当前封装对象。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """离开 with 代码块时释放 DLL 资源。"""
        self.release()

    def last_error(self) -> str:
        """返回最近一次线程局部 DLL 错误信息。"""
        value = self._dll.AI_GetLastError()
        return (value or b"").decode("utf-8", errors="replace")

    def last_latency_us(self, module: int) -> int:
        """返回 CV、OCR 或 YOLO 最近一次测得耗时，单位微秒。"""
        return int(self._dll.AI_GetLastLatencyUs(int(module)))

    def ocr_stage_latency_us(self, stage: int) -> int:
        """返回最近一次 OCR 检测、识别或后处理阶段耗时。"""
        return int(self._dll.AI_GetOcrStageLatencyUs(int(stage)))

    def runtime_status(self) -> Dict[str, object]:
        """返回当前 OCR/全局运行设备状态。"""
        output = C.create_string_buffer(4096)
        self._check(self._dll.AI_GetRuntimeStatusJson(output, len(output)))
        return json.loads(output.value.decode("utf-8"))

    def cv_init(self) -> None:
        """初始化 CV 模块；当前只是轻量就绪检查。"""
        self._check(self._dll.AI_CvInit())

    def cv_context(self) -> "CVContext":
        """创建一个独立的易语言模板缓存实例，可跨线程共享。"""
        return CVContext(self)

    def cv_to_gray(self, image, roi: Optional[Union[AIRect, Sequence[int]]] = None) -> bytes:
        """将图像或 ROI 转为紧凑排列的 8 位灰度字节。"""
        img = _as_image(image)
        rect = _as_roi(roi)
        width = rect.w if rect is not None else img.width
        height = rect.h if rect is not None else img.height
        output = (C.c_uint8 * (width * height))()
        status = self._dll.AI_CvToGray(C.byref(img), C.byref(rect) if rect is not None else None, output, width)
        self._check(status)
        return bytes(output)

    def cv_threshold(
        self,
        image,
        threshold: int,
        roi: Optional[Union[AIRect, Sequence[int]]] = None,
    ) -> bytes:
        """对图像或 ROI 做阈值化，并返回紧凑二值掩码。"""
        img = _as_image(image)
        rect = _as_roi(roi)
        width = rect.w if rect is not None else img.width
        height = rect.h if rect is not None else img.height
        output = (C.c_uint8 * (width * height))()
        status = self._dll.AI_CvThreshold(
            C.byref(img),
            C.byref(rect) if rect is not None else None,
            int(threshold),
            output,
            width,
        )
        self._check(status)
        return bytes(output)

    def cv_extract_trace_json(
        self,
        image,
        roi: Optional[Union[AIRect, Sequence[int]]] = None,
        threshold: int = -1,
        invert: bool = True,
        max_points: int = 4096,
        buffer_size: int = 262144,
    ) -> str:
        """提取前景笔画路径，并返回原始 JSON 字符串。"""
        img = _as_image(image)
        rect = _as_roi(roi)
        size = max(1024, int(buffer_size))
        while True:
            output = C.create_string_buffer(size)
            status = self._dll.AI_CvExtractTraceJson(
                C.byref(img),
                C.byref(rect) if rect is not None else None,
                int(threshold),
                1 if invert else 0,
                int(max_points),
                output,
                size,
            )
            if status == AI_ERR_BUFFER_TOO_SMALL and size < 64 * 1024 * 1024:
                size *= 2
                continue
            self._check(status)
            return output.value.decode("utf-8", errors="replace")

    def cv_extract_trace(
        self,
        image,
        roi: Optional[Union[AIRect, Sequence[int]]] = None,
        threshold: int = -1,
        invert: bool = True,
        max_points: int = 4096,
        buffer_size: int = 262144,
    ) -> Dict[str, object]:
        """提取前景笔画路径，并将 JSON 解析为字典。"""
        return json.loads(
            self.cv_extract_trace_json(
                image,
                roi=roi,
                threshold=threshold,
                invert=invert,
                max_points=max_points,
                buffer_size=buffer_size,
            )
        )

    def cv_mean_color(self, image, roi: Optional[Union[AIRect, Sequence[int]]] = None) -> AIColorStats:
        """计算图像或 ROI 的 BGR 均值和灰度最小/最大值。"""
        img = _as_image(image)
        rect = _as_roi(roi)
        output = AIColorStats()
        self._check(self._dll.AI_CvMeanColor(C.byref(img), C.byref(rect) if rect is not None else None, C.byref(output)))
        return output

    def cv_find_color(
        self,
        image,
        target_bgr: int,
        tolerance: int,
        roi: Optional[Union[AIRect, Sequence[int]]] = None,
    ) -> AIColorFindResult:
        """查找接近指定 BGR 颜色的像素，并返回数量/边界信息。"""
        img = _as_image(image)
        rect = _as_roi(roi)
        output = AIColorFindResult()
        self._check(
            self._dll.AI_CvFindColor(
                C.byref(img),
                C.byref(rect) if rect is not None else None,
                int(target_bgr),
                int(tolerance),
                C.byref(output),
            )
        )
        return output

    def cv_find_image(self, image, template, min_score: float = 0.9) -> Optional[AIImageMatch]:
        """查找单个模板图的最佳出现位置。"""
        img = _as_image(image)
        templ = _as_image(template)
        output = AIImageMatch()
        count = self._check(self._dll.AI_CvFindImage(C.byref(img), C.byref(templ), float(min_score), C.byref(output)))
        return output if count > 0 else None

    def cv_find_image_dict(self, image, template, min_score: float = 0.9) -> Optional[Dict[str, object]]:
        """查找单个模板，并返回便于 JSON 序列化的字典。"""
        match = self.cv_find_image(image, template, min_score)
        return _match_dict(match) if match is not None else None

    def cv_find_images(self, image, templates: Sequence[object], min_score: float = 0.9) -> List[Dict[str, object]]:
        """查找一组不透明模板的所有出现位置。"""
        img = _as_image(image)
        template_images = [_as_image(template) for template in templates]
        if not template_images:
            return []

        template_array = (AIImage * len(template_images))(*template_images)
        raw = self._dll.AI_CvFindImages(C.byref(img), template_array, len(template_images), float(min_score))
        if not raw:
            raise AIEngineError(AI_ERR_RUNTIME, self.last_error())
        return json.loads(raw.decode("utf-8"))

    def cv_find_images_dicts(
        self,
        image,
        templates: Sequence[object],
        min_score: float = 0.9,
    ) -> List[Dict[str, object]]:
        """查找不透明模板，并返回便于 JSON 序列化的字典列表。"""
        return self.cv_find_images(image, templates, min_score)

    def cv_find_transparent_image(
        self,
        image,
        template,
        alpha_threshold: int = 1,
        min_score: float = 0.9,
    ) -> Optional[AIImageMatch]:
        """使用 alpha 掩码查找单个 BGRA/RGBA 模板的最佳出现位置。"""
        img = _as_image(image)
        templ = _as_image(template)
        output = AIImageMatch()
        count = self._check(
            self._dll.AI_CvFindTransparentImage(
                C.byref(img),
                C.byref(templ),
                int(alpha_threshold),
                float(min_score),
                C.byref(output),
            )
        )
        return output if count > 0 else None

    def cv_find_transparent_image_dict(
        self,
        image,
        template,
        alpha_threshold: int = 1,
        min_score: float = 0.9,
    ) -> Optional[Dict[str, object]]:
        """查找单个透明模板，并返回便于 JSON 序列化的字典。"""
        match = self.cv_find_transparent_image(image, template, alpha_threshold, min_score)
        return _match_dict(match) if match is not None else None

    def cv_find_transparent_images(
        self,
        image,
        templates: Sequence[object],
        alpha_threshold: int = 1,
        min_score: float = 0.9,
    ) -> List[Dict[str, object]]:
        """查找一组透明模板的所有出现位置。"""
        img = _as_image(image)
        template_images = [_as_image(template) for template in templates]
        if not template_images:
            return []

        template_array = (AIImage * len(template_images))(*template_images)
        raw = self._dll.AI_CvFindTransparentImages(
            C.byref(img), template_array, len(template_images), int(alpha_threshold), float(min_score)
        )
        if not raw:
            raise AIEngineError(AI_ERR_RUNTIME, self.last_error())
        return json.loads(raw.decode("utf-8"))

    def cv_find_transparent_images_dicts(
        self,
        image,
        templates: Sequence[object],
        alpha_threshold: int = 1,
        min_score: float = 0.9,
    ) -> List[Dict[str, object]]:
        """查找透明模板，并返回便于 JSON 序列化的字典列表。"""
        return self.cv_find_transparent_images(image, templates, alpha_threshold, min_score)

    def has_embedded_assets(self) -> bool:
        """判断当前 DLL 是否内置了默认 YOLO/OCR 模型资源。"""
        return bool(self._dll.AI_HasEmbeddedAssets())

    def yolo_model(
        self,
        input_size: int,
        runtime_device: int,
        device_id: int,
        session_count: int,
    ) -> "YoloModel":
        """创建独立 YOLO 模型句柄；0 自动读取静态模型尺寸，动态模型使用 320 或 640。"""
        return YoloModel(self, input_size, runtime_device, device_id, session_count)

    def ocr_load_embedded_models(self, runtime_device: int = AI_DEVICE_AUTO) -> None:
        """加载 DLL 内置的 PP-OCR 识别模型。"""
        self._check(self._dll.AI_OcrLoadEmbeddedModels(int(runtime_device)))

    def ocr_load_embedded_model(self, runtime_device: int = AI_DEVICE_AUTO, session_count: int = 1, options: Optional[dict[str, object]] = None) -> None:
        """从 DLL 内置 PP-OCRv6 模型创建 OCR 会话池，并可传入性能/检测选项。"""
        if not options:
            self._check(self._dll.OCR_LoadEmbeddedModel(int(runtime_device), int(session_count)))
            return
        value = AIOcrRuntimeOptions(
            int(options.get("det_input_width", 0)),
            int(options.get("det_input_height", 0)),
            int(options.get("intra_op_threads", 0)),
            float(options.get("det_binary_threshold", 0.0)),
            float(options.get("det_box_score_threshold", 0.0)),
            float(options.get("det_unclip_ratio", 0.0)),
        )
        self._check(self._dll.OCR_LoadEmbeddedModelEx(int(runtime_device), int(session_count), C.byref(value)))

    def ocr_recognize_bmp(self, bmp, min_confidence: float, output_format: int = AI_OCR_OUTPUT_TEXT, color_filter: Optional[TextValue] = None, origin_x: int = 0, origin_y: int = 0) -> str:
        """调用易语言兼容 OCR 入口；返回 Windows ACP 文本，TEXT 多行直接拼接。"""
        buf, ptr, size = _bmp_buffer(bmp)
        filter_bytes = _utf8(color_filter) if color_filter not in (None, "", b"") else None
        raw = self._dll.OCR_Recognize(ptr, size, int(output_format), float(min_confidence), filter_bytes, int(origin_x), int(origin_y))
        encoding = "mbcs" if os.name == "nt" else "utf-8"
        return (raw or b"").decode(encoding, errors="strict")

    def ocr_find_multi_text_bmp_text(
        self,
        bmp,
        targets: Union[TextValue, Sequence[TextValue]],
        min_confidence: float,
        color_filter: Optional[TextValue] = None,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> str:
        """原样返回 ID,cx,cy|ID,cx,cy；未命中返回空字符串。"""
        if isinstance(targets, (str, bytes)):
            target_text = targets.decode("utf-8") if isinstance(targets, bytes) else targets
        else:
            target_text = "|".join(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in targets
            )
        buf, ptr, size = _bmp_buffer(bmp)
        encoding = "mbcs" if os.name == "nt" else "utf-8"
        filter_bytes = _utf8(color_filter) if color_filter not in (None, "", b"") else None
        raw = self._dll.OCR_FindMultiText(ptr, size, target_text.encode(encoding), float(min_confidence), filter_bytes, int(origin_x), int(origin_y))
        return (raw or b"").decode(encoding, errors="strict")

    def ocr_find_multi_text_bmp(
        self,
        bmp,
        targets: Union[TextValue, Sequence[TextValue]],
        min_confidence: float,
        color_filter: Optional[TextValue] = None,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> List[Dict[str, object]]:
        """查找多个目标；每项仅包含零基 id 和中心点 cx/cy。"""
        text = self.ocr_find_multi_text_bmp_text(
            bmp, targets, min_confidence, color_filter, origin_x, origin_y)
        if text == "":
            error = self.last_error()
            if error:
                raise AIEngineError(AI_ERR_RUNTIME, error)
            return []
        return _parse_compact_points(text, "cx", "cy")

    def ocr_load_models(
        self,
        det_model_path: Optional[PathLikeValue],
        rec_model_path: Optional[PathLikeValue],
        config_path: Optional[PathLikeValue] = None,
        runtime_device: int = AI_DEVICE_AUTO,
    ) -> None:
        """从磁盘加载可选 OCR 检测模型和必需识别模型。"""
        self._check(
            self._dll.AI_OcrLoadModels(
                _bytes_path(det_model_path),
                _bytes_path(rec_model_path),
                _bytes_path(config_path),
                int(runtime_device),
            )
        )

    def ocr_load_models_from_memory(
        self,
        det_model_data,
        rec_model_data,
        config_path: Optional[PathLikeValue] = None,
        runtime_device: int = AI_DEVICE_AUTO,
    ) -> None:
        """从字节加载 OCR 检测/识别模型。"""
        det_view = memoryview(det_model_data) if det_model_data is not None else None
        rec_view = memoryview(rec_model_data)
        det_buf = (C.c_uint8 * det_view.nbytes).from_buffer_copy(det_view) if det_view is not None and det_view.nbytes > 0 else None
        rec_buf = (C.c_uint8 * rec_view.nbytes).from_buffer_copy(rec_view)
        self._check(
            self._dll.AI_OcrLoadModelsFromMemory(
                C.cast(det_buf, C.c_void_p) if det_buf is not None else None,
                0 if det_buf is None else det_view.nbytes,
                C.cast(rec_buf, C.c_void_p),
                rec_view.nbytes,
                _bytes_path(config_path),
                int(runtime_device),
            )
        )

    def ocr_recognize(self, image, min_confidence: float) -> List[Dict[str, object]]:
        """识别文本，并为每条 OCR 文本行返回一个字典。"""
        img = _as_image(image)
        raw = self._dll.AI_OcrRecognize(C.byref(img), float(min_confidence))
        if not raw:
            raise AIEngineError(AI_ERR_RUNTIME, self.last_error())
        return json.loads(raw.decode("utf-8"))

    def ocr_recognize_lines(
        self,
        image,
        min_confidence: float,
        output_format: int = AI_OCR_OUTPUT_TEXT,
        buffer_size: int = 4096,
    ) -> str:
        """识别文本，并根据 output_format 返回文本或 JSON。"""
        img = _as_image(image)
        output = C.create_string_buffer(int(buffer_size))
        self._check(self._dll.AI_OcrRecognizeLines(C.byref(img), int(output_format), float(min_confidence), output, int(buffer_size)))
        return output.value.decode("utf-8", errors="replace")

    def ocr_recognize_line(
        self,
        image,
        min_confidence: float,
        output_format: int = AI_OCR_OUTPUT_TEXT,
        buffer_size: int = 4096,
    ) -> str:
        """识别文本，并返回合并后的单行结果。"""
        img = _as_image(image)
        output = C.create_string_buffer(int(buffer_size))
        self._check(self._dll.AI_OcrRecognizeLine(C.byref(img), int(output_format), float(min_confidence), output, int(buffer_size)))
        return output.value.decode("utf-8", errors="replace")

    def ocr_find_text(self, image, target: TextValue, min_confidence: float) -> List[Dict[str, object]]:
        """识别文本，并返回 target 匹配项的近似位置框。"""
        img = _as_image(image)
        raw = self._dll.AI_OcrFindText(C.byref(img), _utf8(target), float(min_confidence))
        if not raw:
            raise AIEngineError(AI_ERR_RUNTIME, self.last_error())
        return json.loads(raw.decode("utf-8"))

    def ocr_release(self) -> None:
        """释放当前已加载的 OCR 模型/session。"""
        self._check(self._dll.AI_OcrRelease())

class YoloModel:
    """独立 YOLO 模型和 Session 池；同一对象可由多个 Python 线程共享。"""

    def __init__(self, engine: Engine, input_size: int, runtime_device: int, device_id: int, session_count: int):
        self._engine = engine
        self._input_size = int(input_size)
        self._runtime_device = int(runtime_device)
        self._device_id = int(device_id)
        self._session_count = int(session_count)
        self._handle = C.c_int32(0)
        self._state_lock = threading.RLock()
        self._released = False
        engine._check(engine._dll.AI_YoloCreate(C.byref(self._handle)))

    @property
    def handle(self) -> int:
        return int(self._handle.value)

    def _ensure_open(self) -> int:
        with self._state_lock:
            if self._released:
                raise RuntimeError("YoloModel 已释放")
            return self.handle

    def load_model(
        self,
        model_path: PathLikeValue,
        config_path: Optional[PathLikeValue] = None,
    ) -> None:
        handle = self._ensure_open()
        self._engine._check(
            self._engine._dll.AI_YoloLoadModel(
                handle,
                _bytes_path(model_path),
                _bytes_path(config_path),
                self._input_size,
                self._runtime_device,
                self._device_id,
                self._session_count,
            )
        )

    def load_model_from_memory(
        self,
        model_data,
        config_path: Optional[PathLikeValue] = None,
    ) -> None:
        handle = self._ensure_open()
        view = memoryview(model_data)
        buf = (C.c_uint8 * view.nbytes).from_buffer_copy(view)
        self._engine._check(
            self._engine._dll.AI_YoloLoadModelFromMemory(
                handle,
                C.cast(buf, C.c_void_p),
                view.nbytes,
                _bytes_path(config_path),
                self._input_size,
                self._runtime_device,
                self._device_id,
                self._session_count,
            )
        )

    def load_embedded_model(self) -> None:
        handle = self._ensure_open()
        self._engine._check(
            self._engine._dll.AI_YoloLoadEmbeddedModel(
                handle,
                self._input_size,
                self._runtime_device,
                self._device_id,
                self._session_count,
            )
        )

    def infer(
        self,
        image,
        confidence: float,
    ) -> List[Dict[str, object]]:
        handle = self._ensure_open()
        img = _as_image(image)
        raw = self._engine._dll.AI_YoloInfer(
            handle, C.byref(img), float(confidence)
        )
        if not raw:
            raise AIEngineError(AI_ERR_RUNTIME, self._engine.last_error())
        return json.loads(raw.decode("utf-8"))

    def runtime_status(self) -> Dict[str, object]:
        handle = self._ensure_open()
        buffer = C.create_string_buffer(4096)
        self._engine._check(
            self._engine._dll.YOLO_GetRuntimeStatusJson(handle, buffer, len(buffer))
        )
        return json.loads(buffer.value.decode("utf-8"))

    def last_latency_us(self) -> int:
        return int(self._engine._dll.YOLO_GetLastLatencyUs(self._ensure_open()))

    def release(self) -> None:
        with self._state_lock:
            if self._released:
                return
            self._engine._check(self._engine._dll.AI_YoloRelease(self.handle))
            self._released = True

    def __enter__(self) -> "YoloModel":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class CVContext:
    """易语言模板缓存的显式句柄；一个实例可安全地跨线程共享。"""

    def __init__(self, engine: Engine):
        self._engine = engine
        self._handle = C.c_int32(0)
        self._release_lock = threading.Lock()
        engine._check(engine._dll.CV_Create(C.byref(self._handle)))
        self._released = False

    @property
    def handle(self) -> int:
        return int(self._handle.value)

    def _ensure_open(self) -> int:
        if self._released:
            raise RuntimeError("CVContext 已释放")
        return self.handle

    def load_template_dir(self, directory: PathLikeValue, recursive: bool = False) -> int:
        handle = self._ensure_open()
        return self._engine._check(self._engine._dll.CV_LoadTemplateDir(handle, _compat_text(directory), int(bool(recursive))))

    def load_template_zip(self, zip_data) -> int:
        """从内存标准 ZIP 原子替换模板库；支持 Stored/Deflate BMP。"""
        handle = self._ensure_open()
        buf, ptr, size = _bmp_buffer(zip_data)
        return self._engine._check(
            self._engine._dll.CV_LoadTemplateZipFromMemory(handle, ptr, size)
        )

    def clear(self) -> None:
        self._engine._check(self._engine._dll.CV_ClearTemplateCache(self._ensure_open()))

    def release(self) -> None:
        with self._release_lock:
            if not self._released:
                self._engine._check(self._engine._dll.CV_Release(self.handle))
                self._released = True

    def __enter__(self) -> "CVContext":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _find_one(self, api, template_name, bmp, min_score, match_mode, transparent_rgb=None, origin_x=0, origin_y=0):
        handle = self._ensure_open()
        buf, ptr, size = _bmp_buffer(bmp)
        result = CVMatchResult()
        args = [handle, _compat_text(template_name), ptr, size, float(min_score), int(match_mode)]
        if transparent_rgb is not None:
            args.append(_utf8(_rgb_text(transparent_rgb)))
        args.append(C.byref(result))
        args.extend((int(origin_x), int(origin_y)))
        status = self._engine._check(api(*args))
        return result if status > 0 else None

    def find_one(self, template_name: TextValue, bmp, min_score: float, match_mode: int = 0, origin_x: int = 0, origin_y: int = 0) -> Optional[CVMatchResult]:
        return self._find_one(self._engine._dll.CV_FindOne, template_name, bmp, min_score, match_mode, origin_x=origin_x, origin_y=origin_y)

    def find_transparent_one(self, template_name: TextValue, bmp, min_score: float, transparent_rgb: Union[int, TextValue], match_mode: int = 0, origin_x: int = 0, origin_y: int = 0) -> Optional[CVMatchResult]:
        return self._find_one(self._engine._dll.CV_FindTransparentOne, template_name, bmp, min_score, match_mode, transparent_rgb, origin_x, origin_y)

    def _find_multi(self, api, template_names, bmp, color_bias, min_score, match_mode, transparent_rgb=None, origin_x=0, origin_y=0):
        handle = self._ensure_open()
        buf, ptr, size = _bmp_buffer(bmp)
        args = [handle, _compat_text(template_names), ptr, size, _compat_text(color_bias), float(min_score)]
        if transparent_rgb is not None:
            args.append(_utf8(_rgb_text(transparent_rgb)))
        else:
            args.append(int(match_mode))
        args.extend((int(origin_x), int(origin_y)))
        raw = api(*args)
        if raw is None:
            raise AIEngineError(AI_ERR_RUNTIME, self._engine.last_error())
        encoding = "mbcs" if os.name == "nt" else "utf-8"
        text = raw.decode(encoding, errors="strict")
        if text == "":
            error = self._engine.last_error()
            if error:
                raise AIEngineError(AI_ERR_RUNTIME, error)
            return []
        return _parse_compact_points(text, "x", "y")

    def find_multi(self, template_names: TextValue, bmp, min_score: float, match_mode: int = 0, color_bias: TextValue = b"", origin_x: int = 0, origin_y: int = 0) -> List[Dict[str, object]]:
        """返回仅含 id/x/y 的结构化多模板匹配结果。"""
        return self._find_multi(self._engine._dll.CV_FindMultiText, template_names, bmp, color_bias, min_score, match_mode, origin_x=origin_x, origin_y=origin_y)

    def find_transparent_multi(self, template_names: TextValue, bmp, min_score: float, transparent_rgb: Union[int, TextValue], color_bias: TextValue = b"", origin_x: int = 0, origin_y: int = 0) -> List[Dict[str, object]]:
        """返回仅含 id/x/y 的结构化透明模板匹配结果。"""
        return self._find_multi(self._engine._dll.CV_FindTransparentMultiText, template_names, bmp, color_bias, min_score, match_mode=0, transparent_rgb=transparent_rgb, origin_x=origin_x, origin_y=origin_y)

    def find_multi_text(self, template_names: TextValue, bmp, min_score: float, color_bias: TextValue = b"", match_mode: int = 0, origin_x: int = 0, origin_y: int = 0) -> str:
        """原样返回 ID,x,y|ID,x,y；未命中返回空字符串。"""
        handle = self._ensure_open()
        buf, ptr, size = _bmp_buffer(bmp)
        raw = self._engine._dll.CV_FindMultiText(handle, _compat_text(template_names), ptr, size, _compat_text(color_bias), float(min_score), int(match_mode), int(origin_x), int(origin_y))
        return (raw or b"").decode("mbcs" if os.name == "nt" else "utf-8", errors="strict")

    def find_transparent_multi_text(self, template_names: TextValue, bmp, min_score: float, transparent_rgb: Union[int, TextValue], color_bias: TextValue = b"", origin_x: int = 0, origin_y: int = 0) -> str:
        """原样返回透明匹配的 ID,x,y|ID,x,y；未命中返回空字符串。"""
        handle = self._ensure_open()
        buf, ptr, size = _bmp_buffer(bmp)
        raw = self._engine._dll.CV_FindTransparentMultiText(handle, _compat_text(template_names), ptr, size, _compat_text(color_bias), float(min_score), _rgb_text(transparent_rgb), int(origin_x), int(origin_y))
        return (raw or b"").decode("mbcs" if os.name == "nt" else "utf-8", errors="strict")


__all__ = [
    "AIENGINE_VERSION",
    "AIENGINE_VERSION_MAJOR",
    "AIENGINE_VERSION_MINOR",
    "AIENGINE_VERSION_PATCH",
    "AIEngineError",
    "AIImage",
    "AIRect",
    "AIDetectBox",
    "AIOcrLine",
    "AIOcrRuntimeOptions",
    "AIColorStats",
    "AIColorFindResult",
    "AIImageMatch",
    "CVMatchResult",
    "ImageData",
    "Engine",
    "YoloModel",
    "CVContext",
    "AI_OK",
    "AI_ERR_INVALID_ARGUMENT",
    "AI_ERR_NOT_INITIALIZED",
    "AI_ERR_BACKEND_NOT_CONFIGURED",
    "AI_ERR_IMAGE_FORMAT",
    "AI_ERR_BUFFER_TOO_SMALL",
    "AI_ERR_CONFIG",
    "AI_ERR_INVALID_HANDLE",
    "AI_ERR_ALREADY_LOADED",
    "AI_ERR_BUSY",
    "AI_ERR_RUNTIME",
    "AI_IMAGE_GRAY8",
    "AI_IMAGE_BGR24",
    "AI_IMAGE_BGRA32",
    "AI_IMAGE_RGB24",
    "AI_IMAGE_RGBA32",
    "AI_MODULE_CV",
    "AI_MODULE_OCR",
    "AI_MODULE_YOLO",
    "AI_OCR_STAGE_DETECTION",
    "AI_OCR_STAGE_RECOGNITION",
    "AI_OCR_STAGE_POSTPROCESS",
    "AI_DEVICE_AUTO",
    "AI_DEVICE_DIRECTML",
    "AI_DEVICE_CPU",
    "AI_DEVICE_TENSORRT",
    "AI_OCR_OUTPUT_TEXT",
    "AI_OCR_OUTPUT_JSON",
    "image_from_buffer",
    "image_from_bmp_bytes",
    "image_from_numpy",
]
