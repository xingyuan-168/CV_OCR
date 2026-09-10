#!/usr/bin/env python3
"""Generate and verify the complete public DLL API HTML reference."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path


EXPORT_RE = re.compile(
    r"^AIENGINE_EXPORT[ \t]+(?P<return>[^\r\n]+?)[ \t]+AIENGINE_CALL[ \t]+(?P<name>\w+)[ \t]*\((?P<args>.*?)\)[ \t]*;",
    re.S | re.M,
)

DIRECT_JSON_APIS = {
    "AI_CvFindImages", "AI_CvFindTransparentImages", "AI_OcrRecognize",
    "AI_OcrFindText", "AI_YoloInfer", "AI_YoloDetect",
    "YOLO_InferJson",
}

COMPACT_POINT_APIS = {
    "CV_FindMultiText", "CV_FindTransparentMultiText", "OCR_FindMultiText",
}


def parse_exports(header: Path):
    text = header.read_text(encoding="utf-8")
    exports = []
    for match in EXPORT_RE.finditer(text):
        args = " ".join(match.group("args").split())
        params = [] if args == "void" or not args else [p.strip() for p in args.split(",")]
        exports.append((" ".join(match.group("return").split()), match.group("name"), params))
    return exports


def group(name: str) -> str:
    if name.startswith("CV_"):
        return "易语言 CV 模板找图"
    if name.startswith("OCR_"):
        return "易语言 OCR"
    if name.startswith("YOLO_"):
        return "易语言 YOLO"
    if name.startswith("AI_Cv"):
        return "标准 C ABI：CV"
    if name.startswith("AI_Ocr"):
        return "标准 C ABI：OCR"
    if name.startswith("AI_Yolo"):
        return "标准 C ABI：YOLO"
    return "标准 C ABI：运行时"


def function_summary(name: str) -> str:
    if name == "AI_ShutdownWorker":
        return "请求共享的 CQ_AI_worker.exe 释放全部模型资源并退出；成功返回表示进程已经终止；多个调用方共享 Worker 时，仍在运行的调用方不应随意调用此全局关闭接口。"
    if name in COMPACT_POINT_APIS:
        if name == "OCR_FindMultiText":
            return "查找全部目标文本并返回 ID,cx,cy|ID,cx,cy；ID 保留输入目标列表的零基序号，结果坐标为命中文本中心点。"
        return "查找全部模板目标并返回 ID,x,y|ID,x,y；ID 保留输入模板列表的零基序号，不同模板之间不会互相抑制。"
    if name in DIRECT_JSON_APIS:
        return "执行多结果操作并直接返回 JSON 裸数组；无结果返回 []，不需要结果句柄、容量参数或释放接口。"
    if name == "CV_Create":
        return "创建独立的 CV 模板缓存实例并返回正整数句柄；同一句柄可安全地由多个线程共享。"
    if name == "CV_Release":
        return "释放指定 CV 实例并使句柄立即失效；已开始的查找通过共享快照安全完成。无效或已释放句柄返回参数错误。"
    if name == "CV_ClearTemplateCache":
        return "清空指定 CV 实例的模板集但保留句柄；其他实例不受影响。"
    if name == "CV_LoadTemplateZipFromMemory":
        return "从内存标准 ZIP 加载 Stored/Deflate BMP 模板；全部校验成功后原子替换指定句柄的模板集，调用方式仍使用包内 BMP 文件名。"
    if name in {"YOLO_Release", "AI_YoloRelease"}:
        return "释放指定 YOLO 模型句柄；新调用立即失败，已经开始或正在等待 Session 的调用安全完成。无效或已释放句柄返回 AI_ERR_INVALID_HANDLE。"
    if name in {"AI_Release", "AI_ShutdownWorker", "OCR_Release", "AI_OcrRelease"}:
        return "释放本接口所属资源；全局释放会清理全部 YOLO 模型句柄。"
    if name in {"YOLO_Create", "AI_YoloCreate"}:
        return "创建空 YOLO 模型实例并返回正整数句柄；每个句柄拥有独立模型、运行状态和 Session 池。"
    if name in {"YOLO_GetRuntimeStatusJson", "YOLO_GetLastLatencyUs"}:
        return "按 YOLO 模型句柄查询实际 provider、降级原因、输入尺寸、Session 配置或最近一次推理延迟。"
    if name == "YOLO_InferJson":
        return "执行 YOLO 推理并由 DLL 生成 JSON，再转换为当前 Windows ANSI 代码页文本供易语言直接接收。"
    if name == "OCR_LoadEmbeddedModel":
        return "从 v23.5 单一 Worker 内置的官方 PP-OCRv6 tiny 检测模型、识别模型和字符表创建可配置容量的 OCR Session池。"
    if name == "OCR_LoadEmbeddedModelEx":
        return "从 v23.5 单一 Worker 内置的官方 PP-OCRv6 tiny 模型创建 OCR Session池，并应用检测输入尺寸、ORT线程数和 DB后处理参数。"
    if name == "AI_OcrLoadEmbeddedModels":
        return "从 v23.5 单一 Worker 内置的官方 PP-OCRv6 tiny 检测模型、识别模型和字符表创建默认容量为1的 OCR Session池。"
    if name.startswith(("YOLO_Load", "AI_YoloLoad")):
        return "向空 YOLO 句柄加载一次 ONNX 模型并创建独立 Session 池；成功后禁止在同一句柄重复加载。"
    if "Load" in name or name.endswith("Init"):
        return "创建或替换模型、后端或模板资源。成功后才可调用对应的识别、检测或查找接口。"
    if "Recognize" in name or "FindText" in name:
        if name == "OCR_Recognize":
            return "识别 BMP 中的文本并由 DLL 转换为当前 Windows ANSI 代码页字符串；空滤色时按需执行小字号多行增强，TEXT 多行结果按识别顺序无分隔拼接，JSON 返回原图坐标。"
        if name.startswith("OCR_Find"):
            return "识别 BMP 中的文本并定位目标；内部候选优先选择能命中目标的结果。OCRTextResult 的 x/y 是命中目标子框左上角、cx/cy 是中心点，OCRCoordResult 的 x/y 表示中心点。"
        return "执行 OCR 识别或基于识别结果进行文本定位。min_confidence 是 CTC 非 blank 字符平均置信度的最低接受值。"
    if "Infer" in name or "Detect" in name:
        return "执行当前已加载 YOLO 模型的推理；多结果接口直接返回完整 JSON 数组。"
    if "Find" in name:
        return "执行 CV 图像或颜色查找，并按接口定义返回最佳匹配、全部匹配或结果数组。"
    if name.startswith("AI_Get"):
        return "查询当前 DLL 或 x86 代理连接的 worker 的状态、版本、错误或性能数据。"
    return "执行 ai_engine 的公开功能；详细参数、返回值和资源责任见本页。"


PARAM_TEXT = {
    "handle": "正整数资源句柄。CV 接口使用 CV_Create 返回的模板实例句柄；YOLO 接口使用 YOLO_Create/AI_YoloCreate 返回的模型句柄。句柄可跨线程共享，释放后立即失效。",
    "out_handle": "调用方分配的可写 int32_t 指针，不能为 NULL。创建成功时写入大于 0 且当前进程生命周期内不复用的句柄。",
    "image": "原始像素图像视图 AIImage。data 指向首行像素，width/height 为像素尺寸，stride 可为负数，format 必须是 AI_IMAGE_* 常量。调用期间必须保持像素内存有效。",
    "roi": "可选 AIRect 指针。传空指针表示全图；非空时 x/y/w/h 必须落在输入图像范围内。",
    "output": "调用方分配的可写文本缓冲区或单个结果结构体。函数不会分配该内存；调用完成前必须保持有效。",
    "output_size": "字符输出缓冲区的字节容量，包含结尾空字符。容量不足返回 AI_ERR_BUFFER_TOO_SMALL。",
    "output_stride": "调用方输出图像每行的字节跨度，必须满足接口对 ROI 宽度和像素格式的要求。",
    "output_format": "OCR 文本输出格式：AI_OCR_OUTPUT_TEXT=1 为文本，AI_OCR_OUTPUT_JSON=2 为 JSON。标准 AI_* 接口使用 UTF-8；易语言 OCR_* 兼容接口由 DLL 转为当前 Windows ANSI 代码页。",
    "min_confidence": "OCR CTC 非 blank 字符平均置信度下限，必须为 0.0-1.0。0.0 表示不过滤；低于该值的行不会进入文本、JSON、结构体和坐标查找结果。",
    "color_filter": "可空的单次 OCR 滤色规则。空文本或 NULL 为自动模式；格式为 RRGGBB-RRGGBB，多条规则使用 | 分隔，目标颜色和 RGB 三通道容差均为 6 位十六进制；最多 16 条、总长度 512 字节。非法规则返回 AI_ERR_INVALID_ARGUMENT。",
    "runtime_device": "运行设备：0=AUTO，1=DirectML，2=CPU。AUTO先创建完整DirectML池，失败后销毁未完成池并重建完整CPU池；显式DirectML失败时不整体降级。",
    "device": "运行设备：0=AUTO，1=DirectML，2=CPU。OCR与YOLO含义完全一致；非法值直接返回参数错误。",
    "session_count": "Session 池容量，必须大于 0。YOLO 中它表示同一模型可同时执行的推理数量，不是业务线程总数，也不是 ONNX Runtime 算子内线程数；并发超过容量时等待空闲 Session。GPU 多 Session 会按数量增加显存占用。",
    "options": "仅用于 OCR_LoadEmbeddedModelEx 的 AIOcrRuntimeOptions 指针；允许 NULL 表示全部采用 OCR 自动策略。YOLO 加载接口没有 options 或结构体参数。",
    "input_width": "OCR 检测输入宽度；0 表示使用模型原生尺寸。YOLO 不再分别传输入宽高。",
    "input_height": "OCR 检测输入高度；0 表示使用模型原生尺寸。YOLO 不再分别传输入宽高。",
    "device_id": "DirectML显卡适配器的零基编号，必须大于等于0。DirectML和AUTO严格使用该编号；CPU模式忽略该值但仍应传0。",
    "intra_op_threads": "OCR 高级选项中的单 Session 算子内线程数；0 表示自动。YOLO 不暴露此参数，由程序按逻辑 CPU 数和 Session 数自动计算。",
    "big_data": "完整、未压缩 24 位 BGR BMP 的首地址。不是像素首地址；BMP 文件头和像素数据都必须存在。",
    "big_size": "big_data 指向的完整 BMP 字节数，必须覆盖 BMP 文件头与所有像素行。",
    "det_path": "OCR 检测 ONNX 文件路径。多行识别和文本坐标定位需要检测模型；纯单行识别可传空。相对路径以调用程序 EXE 所在目录为基准。",
    "rec_path": "OCR 识别 ONNX 文件路径，不能为空，且必须与字符表匹配。相对路径以调用程序 EXE 所在目录为基准。",
    "keys_path": "OCR UTF-8 字符表文件路径，不能为空。每行字符必须与识别模型类别顺序一致；相对路径以调用程序 EXE 所在目录为基准。",
    "det_data": "OCR 检测 ONNX 内存首地址；纯识别模式可为 0，非空时需与 det_size 对应。",
    "det_size": "检测模型字节数；没有检测模型时传 0。",
    "det_model_data": "OCR 检测 ONNX 内存首地址；纯识别模式可传空指针，非空时必须与 det_model_size 对应。加载成功前 DLL 会复制数据。",
    "det_model_size": "OCR 检测模型字节数；没有检测模型时传 0。",
    "det_model_path": "可选 OCR 检测 ONNX 文件路径。多行识别与文本定位建议提供；纯识别可传空。",
    "rec_data": "OCR 识别 ONNX 内存首地址，不能为空。DLL/worker 在加载成功前复制数据。",
    "rec_size": "OCR 识别模型字节数，必须大于 0。",
    "rec_model_data": "OCR 识别 ONNX 内存首地址，不能为空。加载成功前 DLL 会复制数据。",
    "rec_model_size": "OCR 识别模型字节数，必须大于 0。",
    "rec_model_path": "OCR 识别 ONNX 文件路径；需要与配置中的字符表兼容，不能为空。",
    "keys_data": "OCR 字符表 UTF-8 内存首地址，不能为空。DLL/worker 在加载成功前复制数据。",
    "keys_size": "OCR 字符表字节数，必须大于 0。",
    "target_utf8": "单个待定位文本，不能为空。标准 AI_* 接口传 UTF-8；易语言 OCR_* 兼容接口传当前 Windows ANSI 代码页文本，DLL 会转换后按 OCR 识别文本做子串查找。",
    "targets_utf8": "多个待定位文本，以 | 分隔；不得以 | 开头或结尾，也不得出现空目标。标准 AI_* 接口传 UTF-8；易语言 OCR_* 兼容接口传当前 Windows ANSI 代码页文本。",
    "template_name": "已通过 CV_LoadTemplateDir 加载的模板名或相对模板路径。",
    "template_names": "多个已加载模板名，以 | 分隔；返回的 template_index/id 为该列表的零基下标。",
    "dir_path": "包含 BMP 模板的目录路径。绝对路径直接使用；相对路径以调用程序 EXE 所在目录为基准，不受当前工作目录影响。成功加载后模板由 DLL 缓存管理，直到 CV_Release 或 CV_ClearTemplateCache。",
    "recursive": "0 仅扫描当前目录；非 0 递归扫描子目录。",
    "zip_data": "完整标准 ZIP 字节首地址。支持 Stored 和 Deflate；可直接传易语言图片与资源中的字节集地址，函数返回前必须保持有效。",
    "zip_size": "ZIP 数据总字节数，必须大于 0。单包最多 4096 项、单项解压 64 MiB、总解压 512 MiB，且拒绝 ZIP64、加密、多卷及异常压缩比。",
    "origin_x": "窗口识别区域左上角 X。仅对成功结果的 x 类坐标做安全加法；可为负数；不需要偏移时传 0。",
    "origin_y": "窗口识别区域左上角 Y。仅对成功结果的 y 类坐标做安全加法；可为负数；不需要偏移时传 0。",
    "min_score": "CV 全分辨率最终精确匹配分数下限，必须为 0.0-1.0。该值只与函数返回的 sim/score 比较，不参与候选粗筛。",
    "match_mode": "CV 颜色匹配模式整数。当前用于保持易语言调用兼容；业务侧应统一配置并实测匹配效果。",
    "transparent_rgb": "RGB 十六进制透明色文本；接受 RRGGBB、#RRGGBB 或 0xRRGGBB，大小写均可。必须恰好表示 24 位颜色；对应模板像素不参与评分。",
    "color_bias": "模板匹配偏色容差文本：两位十六进制为灰度容差，六位十六进制为各通道容差，例如 20 或 101010。",
    "model_path": "YOLO ONNX 模型文件路径，不能为空。绝对路径直接使用；相对路径统一以调用程序 EXE 所在目录为基准，不以 DLL、worker 或当前工作目录为基准。",
    "labels_path": "可选 UTF-8 标签文件路径；传空时结果仅返回类别 ID。绝对/相对路径规则与 model_path 相同。",
    "model_data": "YOLO ONNX 模型内存首地址，不能为空。DLL/worker 在加载成功前复制数据。",
    "model_size": "YOLO ONNX 模型字节数，必须大于 0。",
    "labels_data": "可选 UTF-8 标签文本内存首地址；可与 labels_size 同时为 0。",
    "labels_size": "标签文本字节数；没有标签时传 0。",
    "input_size": "YOLO 方形输入边长。传 0 时自动读取静态 ONNX 的正方形输入尺寸；动态模型必须传 320 或 640。显式尺寸必须匹配静态模型；best.onnx 和 smc.onnx 都可以传 0，加载后通过状态 JSON 查询实际宽高。",
    "conf": "YOLO 置信度阈值，范围 0.0-1.0。",
    "nms": "YOLO 非极大值抑制阈值，范围 0.0-1.0。",
    "module": "延迟模块：AI_MODULE_CV=1、AI_MODULE_OCR=2、AI_MODULE_YOLO=3。",
    "stage": "OCR 阶段：AI_OCR_STAGE_DETECTION=1 检测，AI_OCR_STAGE_RECOGNITION=2 识别，AI_OCR_STAGE_POSTPROCESS=3 后处理。",
    "buffer": "调用方分配的 UTF-8 可写字节缓冲区，用于错误或状态输出。",
    "buffer_size": "buffer 的字节容量，包含结尾空字符。",
    "config_path": "可选 UTF-8 配置文件路径；相对路径以调用程序 EXE 所在目录为基准。配置中的后端、模型和运行时参数会被显式 API 参数覆盖。",
    "model_path": "YOLO ONNX 模型文件路径，不能为空。绝对路径直接使用；相对路径统一以调用程序 EXE 所在目录为基准，不以 DLL、worker 或当前工作目录为基准。",
    "model_data": "模型内存首地址。加载成功前 DLL 会复制数据，之后调用方可释放原内存。",
    "threshold": "图像处理阈值。具体范围由调用接口定义；轨迹提取中 -1 表示自动 Otsu。",
    "invert": "非 0 表示将深色前景视为目标；0 表示浅色前景。",
    "max_points": "轨迹 JSON 最多输出点数；必须按输出缓冲区容量设置。",
    "target_bgr": "目标颜色，低 24 位按 BGR 通道编码。",
    "tolerance": "目标颜色每通道允许误差，必须为非负整数。",
    "templates": "连续 AIImage 模板数组首地址。调用期间每个模板像素内存必须有效。",
    "template_count": "templates 数组元素数，必须大于 0。",
    "templ": "单个 AIImage 模板视图。调用期间其像素内存必须有效。",
    "alpha_threshold": "透明模板有效像素的 alpha 下限，范围 0-255；低于该值的像素忽略。",
}


def param_name(decl: str) -> str:
    cleaned = decl.replace("*", " * ").split()
    return cleaned[-1] if cleaned else decl


def param_text(decl: str, function_name: str) -> str:
    name = param_name(decl)
    compatibility_api = function_name.startswith(("CV_", "OCR_", "YOLO_"))
    if name in {"dir_path", "det_path", "rec_path", "keys_path", "model_path", "labels_path"}:
        path_kind = {
            "dir_path": "CV 模板目录",
            "det_path": "OCR 检测模型",
            "rec_path": "OCR 识别模型",
            "keys_path": "OCR 字符表",
            "model_path": "YOLO ONNX 模型",
            "labels_path": "YOLO 标签",
        }[name]
        optional = "可传空文本；" if name in {"det_path", "labels_path"} else "不能为空；"
        if compatibility_api:
            return f"{path_kind}路径，{optional}易语言按 Windows 当前 ACP 传入，DLL 优先按 ACP 解码；该路径不存在时才兼容尝试合法 UTF-8。绝对路径直接使用，相对路径以调用程序 EXE 所在目录为基准。"
        return f"{path_kind}路径，{optional}标准 AI_* 接口必须传严格 UTF-8。绝对路径直接使用，相对路径以调用程序 EXE 所在目录为基准。"
    if name in {"template_name", "template_names"}:
        plural = "多个模板名以 | 分隔；DLL 会先整串按 ACP 转换再拆分，" if name == "template_names" else "单个模板名或相对模板路径；"
        return f"{plural}易语言 CV_* 接口按 Windows 当前 ACP 传入。ACP 名称未命中时才兼容尝试合法 UTF-8；多模板 ID 使用该输入列表的零基位置。"
    if name == "handle":
        if function_name.startswith(("YOLO_", "AI_Yolo")):
            return "YOLO_Create 或 AI_YoloCreate 返回的正整数模型句柄。同一模型句柄可由多个线程并发共享；释放后失效，不得传 0。"
        return "CV_Create 返回的正整数模板实例句柄。同一句柄可由多个线程并发共享；CV_Release 后失效。"
    if name == "out_handle":
        kind = "YOLO 模型" if function_name.startswith(("YOLO_", "AI_Yolo")) else "CV 模板"
        return f"调用方分配的可写 int32_t 指针，不能为 NULL。成功时写入大于 0 的{kind}句柄，进程生命周期内不复用。"
    if name == "output" and "OCRTextResult" in decl:
        return "调用方分配的 28 字节 OCRTextResult 数值结构体。目标文本已经由调用方传入，因此不重复返回文本；x/y 为左上角，cx/cy 为中心点，w/h 为子框尺寸，score 为识别置信度。调用完成前必须保持内存有效。"
    if name == "output" and "OCRCoordResult" in decl:
        return "调用方分配的 OCRCoordResult 结构体或数组首元素。函数写入命中文本框中心 x/y、框宽高以及 target_index；调用完成前必须保持内存有效。"
    return PARAM_TEXT.get(name, "按函数原型传入该参数。调用方负责其内存有效性、范围合法性与该接口注明的编码要求。")


def return_text(name: str, ret: str) -> str:
    if name == "AI_ShutdownWorker":
        return "AI_OK=0 表示 Worker 已清理并且进程已退出，或 Worker 原本未运行；超时或仍有活动请求时返回负值，详细原因通过 AI_GetLastError() 获取。"
    if name in COMPACT_POINT_APIS:
        shape = "ID,cx,cy|ID,cx,cy" if name == "OCR_FindMultiText" else "ID,x,y|ID,x,y"
        return f"返回 DLL 线程局部持有的 Windows 当前 ANSI 代码页紧凑文本 {shape}；无结果和错误均返回空字符串，错误通过 AI_GetLastError 查询；调用方不得释放。"
    if name in DIRECT_JSON_APIS:
        encoding = "Windows 当前 ANSI 代码页" if name.startswith(("CV_", "OCR_", "YOLO_")) else "UTF-8"
        return f"返回 DLL 线程局部持有的{encoding} JSON 数组；无结果返回 []，错误返回空字符串并通过 AI_GetLastError 查询；调用方不得释放。"
    if name == "CV_Create":
        return "AI_OK=0 表示创建成功且 out_handle 已写入正整数句柄；负值为错误码。"
    if name in {"CV_LoadTemplateDir", "CV_LoadTemplateZipFromMemory"}:
        return "非负值为原子替换后成功加载的 BMP 模板数量；负值为错误码。"
    if name == "CV_Release":
        return "AI_OK=0 表示指定实例已释放；无效或已释放句柄返回 AI_ERR_INVALID_ARGUMENT。"
    if name in {"YOLO_Create", "AI_YoloCreate"}:
        return "AI_OK=0 表示创建成功且 out_handle 已写入正整数模型句柄；负值为错误码。"
    if name in {"YOLO_Release", "AI_YoloRelease"}:
        return "AI_OK=0 表示该模型句柄已从注册表移除；无效或已释放句柄返回 AI_ERR_INVALID_HANDLE。"
    if name == "YOLO_GetLastLatencyUs":
        return "返回该句柄最近一次完成的 BMP 解析、Session 等待和推理总耗时，单位微秒；未推理或句柄无效时返回 -1。"
    if name == "YOLO_GetRuntimeStatusJson":
        return "AI_OK=0 表示 JSON 已写入；无效句柄返回 AI_ERR_INVALID_HANDLE，缓冲区不足返回 AI_ERR_BUFFER_TOO_SMALL。"
    if ret == "void":
        return "无返回值。释放后再次使用对应资源前必须重新加载。"
    if name.startswith("AI_GetVersion"):
        return "返回 DLL 内部持有的只读 UTF-8 版本字符串；调用方不得释放。"
    if name == "AI_GetLastError":
        return "返回 DLL 当前线程内部持有的只读 UTF-8 错误文本；无错误时返回有效空字符串。调用方不传缓冲区或长度，也不得释放。"
    if name.startswith("AI_GetLastLatencyUs"):
        return "返回最近一次指定模块操作的微秒数；没有可用数据时返回 -1。x86 OCR/YOLO 返回 worker 的实际延迟。"
    if name == "AI_GetOcrStageLatencyUs":
        return "返回最近一次 OCR 检测、识别或后处理阶段耗时，单位微秒；没有可用数据时返回 -1。x86 代理返回 worker 对应 session 的阶段耗时。"
    if name.startswith("AI_GetRuntimeStatus"):
        return "返回写入所需字节数（含结尾空字符）；缓冲区不足时返回 AI_ERR_BUFFER_TOO_SMALL。"
    if name == "OCR_Recognize":
        return "返回 DLL 线程局部持有的当前 Windows ANSI 代码页文本或 JSON 指针。TEXT 无结果返回空字符串；JSON 无结果返回 []；参数、BMP、模型或 worker 错误返回空字符串并通过 AI_GetLastError 获取详情。"
    if name == "YOLO_InferJson":
        return "返回 DLL 线程局部持有的 Windows 当前 ANSI 代码页 JSON 文本；无目标返回 []，错误返回空字符串并通过 AI_GetLastError 查询。指针仅保证在同一线程下一次 YOLO_InferJson 调用前有效，调用方不得释放。"
    if name == "OCR_LoadEmbeddedModelEx":
        return "AI_OK=0 表示模型和参数加载成功；负值表示参数、模型、provider 或 worker 错误。静态检测模型收到不支持的输入尺寸时返回 AI_ERR_INVALID_ARGUMENT。"
    if name == "AI_GetOcrStageLatencyUs":
        return "返回大于等于 0 的阶段耗时（微秒）；尚未执行 OCR 或阶段编号无效时返回 -1。"
    if "Release" in name or "Shutdown" in name or "Load" in name or name.endswith("Init"):
        return "AI_OK=0 表示成功；负值为错误码。YOLO 句柄加载成功后重复加载返回 AI_ERR_ALREADY_LOADED。"
    if "Find" in name or "Recognize" in name or "Infer" in name or "Detect" in name:
        return "大于 0 为写入的命中/识别/检测数量；0 为成功但无结果；负值为错误码。"
    return "AI_OK=0 表示成功；负值为错误码。"


def resource_text(name: str) -> str:
    if name == "AI_GetLastError":
        return "返回指针由当前线程的 DLL TLS 持有，不同线程互不覆盖；调用方不得释放，应在同一线程下一次 DLL 接口调用前读取或复制。"
    if name in DIRECT_JSON_APIS:
        return "返回文本由当前调用线程的 DLL TLS 持有，不同线程互不覆盖；调用方不得释放，应在同一线程下一次同类 JSON 调用前解析或复制。"
    if name in COMPACT_POINT_APIS:
        return "返回紧凑文本由当前调用线程的 DLL TLS 持有，不同线程互不覆盖；调用方不得释放，应在同一线程下一次同类多目标调用前解析或复制。"
    if name == "YOLO_InferJson":
        return "返回文本由当前调用线程的 DLL TLS 持有，调用方不得释放，应在同一线程下一次 YOLO_InferJson 前使用或复制。不同线程的返回文本互不覆盖。UTF-8 到 Windows ACP 的转换完全在 DLL 内完成。"
    return "调用方不得在函数执行期间释放输入或输出内存。每个 YOLO 句柄拥有独立 Session 池；同一模型并发超过 session_count 时等待空闲 Session，不同模型的锁和队列互不阻塞。CV 句柄隔离模板集，OCR 保持独立共享池。释放句柄后新调用失败，已开始的调用通过共享所有权安全完成。"


def html_page(exports):
    groups = {}
    for item in exports:
        groups.setdefault(group(item[1]), []).append(item)
    nav = ["<h3>概览</h3><a class='nav-item' onclick=\"show('intro')\">使用与错误码</a>"]
    for title, items in groups.items():
        nav.append(f"<h3>{html.escape(title)}</h3>")
        nav.extend(f"<a class='nav-item' onclick=\"show('{name}')\">{html.escape(name)}</a>" for _, name, _ in items)
    sections = ["""<section id='intro' class='doc-section active'><h1 class='func-title'>CQ_AI 0.14.5（v23.5）公开 DLL 接口说明</h1>
<p>本页由 include/ai_engine.h 生成，并与 CQ_X86.dll 未修饰导出表校验。32位易语言只加载同目录的 CQ_X86.dll；OCR/YOLO统一由 CQ_AI_worker.exe执行。运行目录包含两个运行文件和本 HTML。</p>
<p><b>v23.5 ABI：</b>公开导出保持60个，函数名、参数、返回类型和stdcall参数字节数不变；标准AI_*接口保持不变。Worker协议升级为25。运行设备为0=AUTO、1=DirectML、2=CPU。</p>
<p><b>多目标紧凑文本：</b>CV_FindMultiText和CV_FindTransparentMultiText返回ID,x,y|ID,x,y；OCR_FindMultiText返回ID,cx,cy|ID,cx,cy。ID是输入列表零基序号，不因前项未命中而重排；未命中和错误都返回空文本，错误通过AI_GetLastError区分。</p>
<p><b>当前坐标原点规则：</b>仅命中结果增加坐标起点；三个紧凑多目标接口未命中返回空文本，其他JSON接口未命中返回[]；单结果未命中返回0且输出结构体清零；宽高、分数、文本和序号不变。</p>
<p><b>内存模板包：</b>CV_LoadTemplateZipFromMemory直接读取易语言资源中的标准ZIP，支持Stored/Deflate和BMP；成功后按BMP文件名使用现有找图命令，失败时保留旧缓存。</p>
<p><b>ZIP静态依赖许可：</b>Deflate和CRC使用zlib静态库，不增加运行时DLL。zlib Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler，按zlib License使用；源码许可文本位于 third_party/opencv-5.0.0-build-mt-x86/etc/licenses/zlib-LICENSE。</p>
<p><b>当前 OCR 鲁棒性：</b>显式滤色增加通用抗锯齿恢复候选，候选选择同时评估前景覆盖、尾部漏检、框碎片和置信度；小尺寸单行异常可采用联合行框识别，避免重叠裁剪重复字符。自动模式补充亮字候选。TEXT、JSON和找字接口使用同一最终候选；联合回退时返回原图范围内的联合行框。</p>
<h2>文本与路径编码</h2>
<table class='error-table'><tr><th>入口</th><th>文本和路径输入</th><th>文本返回</th><th>模型附属内容</th></tr>
<tr><td>易语言 CV_* / OCR_* / YOLO_*</td><td>Windows 当前 ACP。DLL 始终先按 ACP 解码；路径不存在或 OCR 目标未命中时，才兼容尝试合法 UTF-8 候选。相对路径以业务 EXE 目录为基准。</td><td>直接文本、紧凑结果和 JSON 为 Windows ACP。</td><td>ONNX 是二进制；内存或文件中的标签、OCR 字符表内容始终为 UTF-8。</td></tr>
<tr><td>标准 AI_* C API</td><td>严格 UTF-8。</td><td>UTF-8。</td><td>ONNX 是二进制；标签和 OCR 字符表内容为 UTF-8。</td></tr></table>
<p>CV 多模板名称先将完整 ACP 字符串转换为 UTF-8，再按 | 分隔，避免 CP936 双字节字符被误拆。易语言调用方不要自行把中文参数转换成 UTF-8。</p>
<h2>OCR 阅读顺序与内部裁剪</h2>
<p>OCR 检测框先按阅读顺序归行：行从上到下，同行从左到右。TEXT 按该顺序直接拼接；JSON lines 使用相同顺序并保留原始检测框坐标。识别模型的内部裁剪会向文本框上方补充一个像素以保留顶部笔画，但不会改变公开 x/y/w/h。</p>
<h2>YOLO 加载参数填写规范</h2>
<table class='error-table'><tr><th>参数</th><th>必须填写的规则</th></tr>
<tr><td>模型路径 / 标签路径</td><td>模型路径不能为空；标签路径可传空。绝对路径直接使用，相对路径以调用程序 EXE 所在目录为基准。DLL 和 worker 不固定模型目录。</td></tr>
<tr><td>模型尺寸</td><td>填写 0 时由 DLL 自动读取静态 ONNX 的方形输入尺寸；动态模型必须填写 320 或 640。显式尺寸必须与模型输入一致。best.onnx 和 smc.onnx 都可填写 0，加载后通过 YOLO_GetRuntimeStatusJson 查询实际宽高。</td></tr>
<tr><td>运行设备</td><td>0=AUTO，1=DirectML，2=CPU。AUTO先创建完整DirectML Session池，失败后销毁未完成池并重建完整CPU池；显式DirectML失败时不整体降级。</td></tr>
<tr><td>GPU 设备序号</td><td>从0开始且不能为负数。DirectML和AUTO使用该DXGI适配器序号；CPU模式忽略但仍填写0。软件适配器、越界或不支持DX12会返回具体错误。</td></tr>
<tr><td>并发线程数</td><td>必须大于 0，实际表示该模型 Session 池容量，不是 ORT 内部线程数。GPU 多 Session 会增加显存占用。每 Session 内部线程由程序计算：min(4, max(1, 逻辑 CPU 数 / Session 数))。</td></tr></table>
<p>路径加载示例：YOLO_LoadModelFromPath(句柄, best路径, 空标签, 0, AI_DEVICE_AUTO, 0, 并发数)；另一个句柄加载 smc.onnx 时也可填写 0 自动识别。内存加载使用相同四项运行参数。NMS 由 worker 内部统一处理，调用方不传。加载失败后立即调用 AI_GetLastError 获取具体原因，成功后调用 YOLO_GetRuntimeStatusJson 核对实际尺寸、provider、设备序号、Session 数和内部线程数。</p>
<h2>通用错误码</h2><table class='error-table'><tr><th>值</th><th>含义</th></tr><tr><td>-1</td><td>参数错误</td></tr><tr><td>-2</td><td>未初始化</td></tr><tr><td>-3</td><td>后端或模型未配置</td></tr><tr><td>-4</td><td>图像格式错误</td></tr><tr><td>-5</td><td>输出缓冲区不足</td></tr><tr><td>-6</td><td>配置或模型创建失败</td></tr><tr><td>-7</td><td>模型或实例句柄无效</td></tr><tr><td>-8</td><td>YOLO 句柄已经成功加载模型</td></tr><tr><td>-9</td><td>模型正在加载或资源忙</td></tr><tr><td>-100</td><td>运行时或 worker 错误</td></tr></table></section>"""]
    for ret, name, params in exports:
        proto = f"{ret} __stdcall {name}({', '.join(params) if params else 'void'});"
        rows = "<tr><td>无</td><td>无参数。</td></tr>" if not params else "".join(
            f"<tr><td class='param-name'>{html.escape(param_name(p))}</td><td><code>{html.escape(p)}</code><br>{html.escape(param_text(p, name))}</td></tr>" for p in params)
        sections.append(
            f"<section id='{name}' class='doc-section'><h1 class='func-title'>{name}</h1><h2>函数简介</h2><p>{html.escape(function_summary(name))}</p><h2>适用范围</h2><p>{'CQ_X86.dll 公开易语言 ABI；文本型参数按 Windows ACP 输入，OCR/YOLO 由同目录的 v23.5 单一 Worker 执行。' if name.startswith(('CV_', 'OCR_', 'YOLO_')) else '标准 C ABI；文本与路径参数严格使用 UTF-8；v23.5 正式包不包含 x64 业务 DLL。'}</p><h2>函数原型</h2><pre>{html.escape(proto)}</pre><h2>参数定义</h2><table class='param-list'>{rows}</table><h2>返回值</h2><table class='error-table'><tr><th>返回值</th><th>说明</th></tr><tr><td>{html.escape(ret)}</td><td>{html.escape(return_text(name, ret))}</td></tr></table><h2>资源与并发</h2><p>{html.escape(resource_text(name))}</p></section>"
        )
    head = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CQ_AI 易语言 DLL 接口说明</title><style>body{font-family:SimSun,'Microsoft YaHei',sans-serif;margin:0;display:flex;height:100vh;font-size:14px}#sidebar{width:280px;overflow:auto;background:#f0f0f0;border-right:1px solid #ccc;padding:10px 0;flex-shrink:0}#sidebar h3{margin:12px 10px 5px;font-size:14px}.nav-item{display:block;padding:4px 10px 4px 25px;color:#000;text-decoration:none;cursor:pointer;font-size:13px}.nav-item:hover{background:#ddeeff;color:#06c}.nav-item.active{background:#3399ff;color:#fff}#content{padding:20px 30px;overflow:auto;flex:1}.doc-section{display:none;max-width:900px}.doc-section.active{display:block}h1.func-title{font-size:20px;border-bottom:1px solid #ddd;padding-bottom:5px}h2{font-size:16px;color:#03c;margin:15px 0 8px}p{margin:5px 0 5px 20px;line-height:1.6}pre{background:#f9f9f9;border:1px solid #eee;padding:10px;margin-left:20px;white-space:pre-wrap}.param-list,.error-table{border-collapse:collapse;margin-left:20px;width:92%}.param-list td,.error-table td,.error-table th{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}.error-table th{background:#f5f5f5}.param-name{font-weight:bold;color:#a52a2a;white-space:nowrap}code{font-family:Consolas,monospace}</style></head><body><div id='sidebar'>"""
    foot = """</div><script>function show(id){document.querySelectorAll('.doc-section').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');location.hash=id}addEventListener('load',()=>{const id=location.hash.slice(1);if(id&&document.getElementById(id))show(id)})</script></body></html>"""
    return head + "".join(nav) + "</div><div id='content'>" + "".join(sections) + foot


def exported_names(dumpbin: Path, dll: Path):
    result = subprocess.run([str(dumpbin), "/exports", str(dll)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    names = set()
    for line in result.stdout.splitlines():
        match = re.match(r"\s*\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)$", line)
        if match and not match.group(1).startswith("_"):
            names.add(match.group(1))
    return names


def module_declarations(module: Path):
    declarations = {}
    current = None
    count = 0
    for raw in module.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(".DLL命令"):
            if current is not None:
                declarations[current] = count
            quoted = re.findall(r'"([^"]+)"', line)
            current = quoted[-1] if len(quoted) >= 2 and Path(quoted[-2]).name.lower() == "cq_x86.dll" else None
            count = 0
        elif current is not None and line.startswith(".参数"):
            count += 1
    if current is not None:
        declarations[current] = count
    return declarations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, default=Path("include/ai_engine.h"))
    parser.add_argument("--output", type=Path, default=Path("docs/易语言_DLL_API_说明.html"))
    parser.add_argument("--dumpbin", type=Path)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--module", type=Path)
    args = parser.parse_args()
    exports = parse_exports(args.header)
    if not exports or any("#define" in ret or "AIENGINE_EXPORT" in ret for ret, _, _ in exports):
        raise SystemExit("header parser produced an invalid export declaration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_page(exports), encoding="utf-8")
    doc = args.output.read_text(encoding="utf-8")
    missing_sections = [name for _, name, _ in exports if f"id='{name}'" not in doc]
    if missing_sections:
        raise SystemExit("missing documentation sections: " + ", ".join(missing_sections))
    if args.dumpbin and args.dll:
        actual = exported_names(args.dumpbin, args.dll)
        expected = {name for _, name, _ in exports}
        if actual != expected:
            raise SystemExit("header/export mismatch\nmissing=" + repr(sorted(expected - actual)) + "\nunexpected=" + repr(sorted(actual - expected)))
    if args.module:
        expected_counts = {name: len(params) for _, name, params in exports}
        declarations = module_declarations(args.module)
        unknown = sorted(set(declarations) - set(expected_counts))
        mismatched = sorted(
            name for name, count in declarations.items()
            if name in expected_counts and expected_counts[name] != count
        )
        if unknown or mismatched:
            details = ["module/header mismatch"]
            if unknown:
                details.append("unknown=" + repr(unknown))
            if mismatched:
                details.append("parameter_counts=" + repr([
                    (name, declarations[name], expected_counts[name]) for name in mismatched
                ]))
            raise SystemExit("\n".join(details))
    print(f"generated {args.output} for {len(exports)} public exports")


if __name__ == "__main__":
    main()
