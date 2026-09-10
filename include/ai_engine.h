#pragma once

#include <stdint.h>

// ai_engine.dll 的公开 C ABI。
//
// 该 DLL 需要同时给 Python ctypes、易语言、C/C++ 调用，因此这里刻意只使用
// C 兼容结构体、整数常量和 stdcall 导出，避免公开 C++ 类型。
#if defined(_WIN32)
#define AIENGINE_CALL __stdcall
#define AIENGINE_EXPORT extern "C" __declspec(dllexport)
#else
#define AIENGINE_CALL
#define AIENGINE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define AIENGINE_VERSION_MAJOR 0
#define AIENGINE_VERSION_MINOR 14
#define AIENGINE_VERSION_PATCH 5

#define AIENGINE_MAX_LABEL 32
#define AIENGINE_MAX_TEXT 512

// 所有 int32_t 接口共用的状态码。
// 非负值表示成功；识别、搜索类接口通常返回正数数量。负数可配合
// AI_GetLastError() 获取更详细错误信息。
enum AIEngineStatus {
    AI_OK = 0,
    AI_ERR_INVALID_ARGUMENT = -1,
    AI_ERR_NOT_INITIALIZED = -2,
    AI_ERR_BACKEND_NOT_CONFIGURED = -3,
    AI_ERR_IMAGE_FORMAT = -4,
    AI_ERR_BUFFER_TOO_SMALL = -5,
    AI_ERR_CONFIG = -6,
    AI_ERR_INVALID_HANDLE = -7,
    AI_ERR_ALREADY_LOADED = -8,
    AI_ERR_BUSY = -9,
    AI_ERR_RUNTIME = -100
};

// AIImage::format 使用的像素排列。BMP 截图通常是 AI_IMAGE_BGR24 或 AI_IMAGE_BGRA32。
enum AIImageFormat {
    AI_IMAGE_GRAY8 = 1,
    AI_IMAGE_BGR24 = 2,
    AI_IMAGE_BGRA32 = 3,
    AI_IMAGE_RGB24 = 4,
    AI_IMAGE_RGBA32 = 5
};

// AI_GetLastLatencyUs() 使用的模块编号。
enum AIModule {
    AI_MODULE_CV = 1,
    AI_MODULE_OCR = 2,
    AI_MODULE_YOLO = 3
};

enum AIOcrLatencyStage {
    AI_OCR_STAGE_DETECTION = 1,
    AI_OCR_STAGE_RECOGNITION = 2,
    AI_OCR_STAGE_POSTPROCESS = 3
};

// ONNX Runtime YOLO/OCR 会话使用的运行设备选择。
// AUTO 先创建完整 DirectML Session 池，失败后重新创建完整 CPU Session 池。
enum AIRuntimeDevice {
    AI_DEVICE_AUTO = 0,
    AI_DEVICE_DIRECTML = 1,
    AI_DEVICE_CPU = 2
};

// AI_OcrRecognizeLine(s) 文本辅助接口的输出格式。
enum AIOcrOutputFormat {
    AI_OCR_OUTPUT_TEXT = 1,
    AI_OCR_OUTPUT_JSON = 2
};

#pragma pack(push, 1)

// 传入所有 CV/YOLO/OCR 函数的原始图像视图。
//
// data 指向逻辑上的第一行；stride 表示每行字节数，可以为负数，便于直接引用
// 自底向上的 BMP 像素缓冲区。
typedef struct AIImage {
    uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t format;
} AIImage;

// 图像坐标中的矩形。CV 接口传入空 AIRect* 表示整张图。
typedef struct AIRect {
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
} AIRect;

// 单个 YOLO 检测结果，坐标为原图坐标。
typedef struct AIDetectBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int32_t class_id;
    char label[AIENGINE_MAX_LABEL];
} AIDetectBox;

// 单个 OCR 文本行/区域结果。text 为 UTF-8，始终以空字符结尾。
typedef struct AIOcrLine {
    AIRect box;
    float confidence;
    char text[AIENGINE_MAX_TEXT];
} AIOcrLine;

// OCR 内置模型运行参数。数值为 0 时由 DLL 根据模型和 CPU 自动选择。
// det_input_width/height 只有在检测 ONNX 支持动态输入或该尺寸与模型一致时才会生效。
typedef struct AIOcrRuntimeOptions {
    int32_t det_input_width;
    int32_t det_input_height;
    int32_t intra_op_threads;
    float det_binary_threshold;
    float det_box_score_threshold;
    float det_unclip_ratio;
} AIOcrRuntimeOptions;

// ROI 内的 BGR 均值和灰度范围。
typedef struct AIColorStats {
    double mean_b;
    double mean_g;
    double mean_r;
    double min_gray;
    double max_gray;
    int32_t pixels;
} AIColorStats;

// AI_CvFindColor() 的结果摘要。
typedef struct AIColorFindResult {
    int32_t count;
    float ratio;
    int32_t first_x;
    int32_t first_y;
    AIRect bounds;
} AIColorFindResult;

// 模板匹配结果。template_index 仅在多模板接口中有意义。
typedef struct AIImageMatch {
    AIRect box;
    float score;
    int32_t template_index;
} AIImageMatch;

// 易语言友好 CV 找图结果。坐标均为左上角原点。
typedef struct CVMatchResult {
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
    float sim;
    int32_t template_index;
} CVMatchResult;

// 易语言友好 OCR 找字结果。x/y 为命中目标子框左上角，cx/cy 为该子框中心点。
// 目标文本已经由调用方传入，因此单文本结果只返回坐标、尺寸和置信度。
typedef struct OCRTextResult {
    int32_t cx;
    int32_t cy;
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
    float score;
} OCRTextResult;

// 易语言友好 OCR 坐标结果。x/y 为命中目标子框中心点；多目标查找时
// target_index 对应输入 "目标1|目标2" 的零基序号。
typedef struct OCRCoordResult {
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
    int32_t target_index;
} OCRCoordResult;

#pragma pack(pop)

// 返回 DLL 内部持有的静态 UTF-8 版本字符串。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_GetVersion(void);

// 初始化全局引擎，默认自动选择 DirectML，失败时回退 CPU。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_Init(const char* config_path);

// 初始化全局引擎，并用 runtime_device 覆盖 runtime.device。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_InitEx(
    const char* config_path,
    int32_t runtime_device);

// 释放全局引擎以及已加载的 YOLO/OCR 模型状态。
AIENGINE_EXPORT void AIENGINE_CALL AI_Release(void);

// 释放并终止 x86 易语言代理自动创建的 x64 worker；成功返回前等待 worker 进程真正退出。
// worker 未运行时返回 AI_OK；多个调用方共享 worker 时这是全局关闭请求。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_ShutdownWorker(void);

// 返回当前调用线程最近一次错误的只读 UTF-8 文本；调用方不得释放。
// 指针有效到同一线程下一次 DLL 接口调用；没有错误时返回有效空字符串。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_GetLastError(void);

// 返回 AI_MODULE_CV/OCR/YOLO 最近一次操作耗时，单位微秒。
AIENGINE_EXPORT int64_t AIENGINE_CALL AI_GetLastLatencyUs(int32_t module);

// 返回最近一次 OCR 检测、识别或后处理阶段耗时，单位微秒；无数据时返回 -1。
AIENGINE_EXPORT int64_t AIENGINE_CALL AI_GetOcrStageLatencyUs(int32_t stage);

// 将当前进程最近一次模型加载选择的执行 provider 写为 JSON。
// 输出示例：{"requested":"auto","active":"cpu","degraded":true,"reason":"..."}
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_GetRuntimeStatusJson(char* buffer, int32_t buffer_size);

// 判断当前 DLL 是否已经内置默认 YOLO/OCR 模型、标签和字典资源；返回 1 表示存在，0 表示不存在。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_HasEmbeddedAssets(void);

// 创建空 YOLO 模型实例。句柄成功加载后可被多个线程并发共享。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloCreate(int32_t* out_handle);

// 从 DLL 内置资源加载默认 YOLO 模型。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadEmbeddedModel(
    int32_t handle,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count);

// 从 DLL 内置资源加载默认 PP-OCR 识别模型，不需要传入模型路径或配置路径。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadEmbeddedModels(int32_t runtime_device);

// AI_YoloInfer() 的别名；返回线程局部 UTF-8 JSON 裸数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_YoloDetect(
    int32_t handle,
    const AIImage* image,
    float conf);

// 从路径加载 YOLO ONNX 模型。config_path 可为空；模型尺寸、设备、设备序号和 Session 数量显式提供。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadModel(
    int32_t handle,
    const char* model_path,
    const char* config_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count);

// 从内存加载 YOLO ONNX 模型。DLL 复制一次模型字节，成功后调用方可以释放原缓冲区。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadModelFromMemory(
    int32_t handle,
    const void* model_data,
    int32_t model_size,
    const char* config_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count);

// 使用指定模型执行 YOLO 推理；conf 为 0.0-1.0，NMS 由内部统一处理；返回线程局部 UTF-8 JSON 裸数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_YoloInfer(
    int32_t handle,
    const AIImage* image,
    float conf);

// 释放指定 YOLO 模型实例。无效或已释放句柄返回 AI_ERR_INVALID_HANDLE。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloRelease(int32_t handle);

// 执行 OCR，并返回全部文本行组成的线程局部 UTF-8 JSON 裸数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_OcrRecognize(
    const AIImage* image,
    float min_confidence);

// 从路径加载 OCR 检测/识别模型。
// 纯识别模式下 det_model_path 可以为空。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadModels(
    const char* det_model_path,
    const char* rec_model_path,
    const char* config_path,
    int32_t runtime_device);

// 从内存加载 OCR 模型。rec_model_data 必填；纯识别模式下 det_model_data 可为空/0。
// DLL 会复制传入的模型字节。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadModelsFromMemory(
    const void* det_model_data,
    int32_t det_model_size,
    const void* rec_model_data,
    int32_t rec_model_size,
    const char* config_path,
    int32_t runtime_device);

// 释放当前已加载的 OCR 模型状态。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRelease(void);

// 识别并将第一条 OCR 文本行格式化为 UTF-8 文本或 JSON。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRecognizeLine(
    const AIImage* image,
    int32_t output_format,
    float min_confidence,
    char* output,
    int32_t output_size);

// 识别全部 OCR 文本行，并格式化为 UTF-8 文本或 JSON。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRecognizeLines(
    const AIImage* image,
    int32_t output_format,
    float min_confidence,
    char* output,
    int32_t output_size);

// 先识别文本，再以线程局部 UTF-8 JSON 裸数组返回 target_utf8 的全部出现位置。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_OcrFindText(
    const AIImage* image,
    const char* target_utf8,
    float min_confidence);

// 轻量 CV 初始化入口。当前无状态，固定返回 AI_OK。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvInit(void);

// 将 ROI 转为 8 位灰度图，输出每行跨度为 output_stride 字节。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvToGray(
    const AIImage* image,
    const AIRect* roi,
    uint8_t* output,
    int32_t output_stride);

// 使用 [0,255] 范围内阈值，将 ROI 转为 8 位二值图。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvThreshold(
    const AIImage* image,
    const AIRect* roi,
    int32_t threshold,
    uint8_t* output,
    int32_t output_stride);

// 将前景笔画中心线提取为 UTF-8 JSON。
// threshold：0-255；传 -1 表示自动使用 Otsu 阈值。
// invert：非 0 表示浅色背景上的深色前景。
// max_points：所有路径最多序列化的点数；<=0 使用默认值。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvExtractTraceJson(
    const AIImage* image,
    const AIRect* roi,
    int32_t threshold,
    int32_t invert,
    int32_t max_points,
    char* output,
    int32_t output_size);

// 计算 ROI 内 B/G/R 均值以及灰度最小/最大值。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvMeanColor(
    const AIImage* image,
    const AIRect* roi,
    AIColorStats* output);

// 查找接近 target_bgr 的像素。target_bgr 位 0-7=B、8-15=G、16-23=R；
// tolerance 按单通道分别比较。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindColor(
    const AIImage* image,
    const AIRect* roi,
    uint32_t target_bgr,
    int32_t tolerance,
    AIColorFindResult* output);

// 查找最佳非透明模板匹配。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindImage(
    const AIImage* image,
    const AIImage* templ,
    float min_score,
    AIImageMatch* output);

// 在模板数组中查找多个非透明模板匹配，并返回线程局部 UTF-8 JSON 裸数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_CvFindImages(
    const AIImage* image,
    const AIImage* templates,
    int32_t template_count,
    float min_score);

// 查找最佳透明模板匹配，忽略 alpha 低于阈值的模板像素。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindTransparentImage(
    const AIImage* image,
    const AIImage* templ,
    int32_t alpha_threshold,
    float min_score,
    AIImageMatch* output);

// 在模板数组中查找多个透明模板匹配，并返回线程局部 UTF-8 JSON 裸数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_CvFindTransparentImages(
    const AIImage* image,
    const AIImage* templates,
    int32_t template_count,
    int32_t alpha_threshold,
    float min_score);

// ---------------------------------------------------------------------------
// 易语言友好接口：文本型参数优先按 Windows 当前 ACP 解码；路径相对于业务 EXE。
// 路径的 ACP 候选不存在时才兼容尝试合法 UTF-8。模型、标签和字符表内存内容仍为 UTF-8。
// 输入大图为大漠 GetScreenDataBmp 返回的 24 位 BMP 内存。
// ---------------------------------------------------------------------------

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_Create(int32_t* out_handle);
AIENGINE_EXPORT int32_t AIENGINE_CALL CV_LoadTemplateDir(int32_t handle, const char* dir_path, int32_t recursive);
// 从标准 ZIP 内存加载 Stored/Deflate BMP 模板；全部成功后原子替换当前模板集。
AIENGINE_EXPORT int32_t AIENGINE_CALL CV_LoadTemplateZipFromMemory(
    int32_t handle,
    const uint8_t* zip_data,
    int32_t zip_size);
AIENGINE_EXPORT int32_t AIENGINE_CALL CV_ClearTemplateCache(int32_t handle);
AIENGINE_EXPORT int32_t AIENGINE_CALL CV_Release(int32_t handle);

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_FindOne(
    int32_t handle,
    const char* template_name,
    const uint8_t* big_data,
    int32_t big_size,
    float min_score,
    int32_t match_mode,
    CVMatchResult* output,
    int32_t origin_x,
    int32_t origin_y);

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_FindTransparentOne(
    int32_t handle,
    const char* template_name,
    const uint8_t* big_data,
    int32_t big_size,
    float min_score,
    int32_t match_mode,
    const char* transparent_rgb,
    CVMatchResult* output,
    int32_t origin_x,
    int32_t origin_y);

// 易语言紧凑多目标结果：ID,x,y|ID,x,y；空结果返回空文本。
// 返回指针仅在当前线程的下一次同类调用前有效。
AIENGINE_EXPORT const char* AIENGINE_CALL CV_FindMultiText(
    int32_t handle,
    const char* template_names,
    const uint8_t* big_data,
    int32_t big_size,
    const char* color_bias,
    float min_score,
    int32_t match_mode,
    int32_t origin_x,
    int32_t origin_y);

AIENGINE_EXPORT const char* AIENGINE_CALL CV_FindTransparentMultiText(
    int32_t handle,
    const char* template_names,
    const uint8_t* big_data,
    int32_t big_size,
    const char* color_bias,
    float min_score,
    const char* transparent_rgb,
    int32_t origin_x,
    int32_t origin_y);

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadModelFromPath(
    const char* det_path,
    const char* rec_path,
    const char* keys_path,
    int32_t device,
    int32_t session_count);

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadModelFromMemory(
    const void* det_data,
    int32_t det_size,
    const void* rec_data,
    int32_t rec_size,
    const void* keys_data,
    int32_t keys_size,
    int32_t device,
    int32_t session_count);

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadEmbeddedModel(
    int32_t device,
    int32_t session_count);

// 使用 DLL 内置 PP-OCRv6 模型和可选性能/检测参数加载 OCR session 池。
// options 传 NULL 等价于 OCR_LoadEmbeddedModel；DLL 会复制/读取参数后再返回。
AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadEmbeddedModelEx(
    int32_t device,
    int32_t session_count,
    const AIOcrRuntimeOptions* options);

AIENGINE_EXPORT const char* AIENGINE_CALL OCR_Recognize(
    const uint8_t* big_data,
    int32_t big_size,
    int32_t output_format,
    float min_confidence,
    // 可空；空文本/NULL 为自动模式；RRGGBB-RRGGBB，多条使用 | 分隔。
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y);

// 下列 OCR_* 兼容入口的 target_utf8/targets_utf8 参数名为兼容旧头文件保留；
// 易语言调用时实际传入当前 Windows ANSI 代码页文本，DLL 内部会转换为 UTF-8。
AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_FindOneText(
    const uint8_t* big_data,
    int32_t big_size,
    const char* target_utf8,
    float min_confidence,
    OCRTextResult* output,
    // 可空；滤色仅影响本次识别，不改变 OCR 模型池。
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y);

// 返回 ID,cx,cy|ID,cx,cy；ID 为输入目标的零基序号；空结果返回空文本。
AIENGINE_EXPORT const char* AIENGINE_CALL OCR_FindMultiText(
    const uint8_t* big_data,
    int32_t big_size,
    const char* targets_utf8,
    float min_confidence,
    // 可空；格式为 RRGGBB-RRGGBB，多条规则按 OR 合并。
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y);

// 仅返回单个目标的坐标。成功返回 1，未命中返回 0。
AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_FindOneCoord(
    const uint8_t* big_data,
    int32_t big_size,
    const char* target_utf8,
    float min_confidence,
    OCRCoordResult* output,
    // 可空；坐标结果映射回原始 BMP。
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y);

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_Release(void);

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_Create(int32_t* out_handle);

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_LoadModelFromPath(
    int32_t handle,
    const char* model_path,
    const char* labels_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count);

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_LoadModelFromMemory(
    int32_t handle,
    const void* model_data,
    int32_t model_size,
    const void* labels_data,
    int32_t labels_size,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count);

AIENGINE_EXPORT const char* AIENGINE_CALL YOLO_InferJson(
    int32_t handle,
    const uint8_t* big_data,
    int32_t big_size,
    float conf,
    int32_t origin_x,
    int32_t origin_y);

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_GetRuntimeStatusJson(
    int32_t handle,
    char* output,
    int32_t output_size);

AIENGINE_EXPORT int64_t AIENGINE_CALL YOLO_GetLastLatencyUs(int32_t handle);
AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_Release(int32_t handle);

#if defined(__cplusplus)
// C++ source-compatibility helpers.  The exported stdcall ABI always contains
// the two origin integers; these overloads only make existing C++ callers pass
// the required zero defaults explicitly at compile time.
inline int32_t CV_FindOne(
    int32_t handle, const char* template_name, const uint8_t* big_data,
    int32_t big_size, float min_score, int32_t match_mode, CVMatchResult* output) {
    return ::CV_FindOne(
        handle, template_name, big_data, big_size, min_score, match_mode, output, 0, 0);
}

inline int32_t CV_FindTransparentOne(
    int32_t handle, const char* template_name, const uint8_t* big_data,
    int32_t big_size, float min_score, int32_t match_mode,
    const char* transparent_rgb, CVMatchResult* output) {
    return ::CV_FindTransparentOne(
        handle, template_name, big_data, big_size, min_score, match_mode,
        transparent_rgb, output, 0, 0);
}

inline const char* CV_FindMultiText(
    int32_t handle, const char* template_names, const uint8_t* big_data,
    int32_t big_size, const char* color_bias, float min_score, int32_t match_mode) {
    return ::CV_FindMultiText(
        handle, template_names, big_data, big_size, color_bias, min_score, match_mode, 0, 0);
}

inline const char* CV_FindTransparentMultiText(
    int32_t handle, const char* template_names, const uint8_t* big_data,
    int32_t big_size, const char* color_bias, float min_score,
    const char* transparent_rgb) {
    return ::CV_FindTransparentMultiText(
        handle, template_names, big_data, big_size, color_bias, min_score,
        transparent_rgb, 0, 0);
}

inline const char* OCR_Recognize(
    const uint8_t* big_data, int32_t big_size, int32_t output_format,
    float min_confidence, const char* color_filter) {
    return ::OCR_Recognize(
        big_data, big_size, output_format, min_confidence, color_filter, 0, 0);
}

inline int32_t OCR_FindOneText(
    const uint8_t* big_data, int32_t big_size, const char* target_utf8,
    float min_confidence, OCRTextResult* output, const char* color_filter) {
    return ::OCR_FindOneText(
        big_data, big_size, target_utf8, min_confidence, output, color_filter, 0, 0);
}

inline const char* OCR_FindMultiText(
    const uint8_t* big_data, int32_t big_size, const char* targets_utf8,
    float min_confidence, const char* color_filter) {
    return ::OCR_FindMultiText(
        big_data, big_size, targets_utf8, min_confidence, color_filter, 0, 0);
}

inline int32_t OCR_FindOneCoord(
    const uint8_t* big_data, int32_t big_size, const char* target_utf8,
    float min_confidence, OCRCoordResult* output, const char* color_filter) {
    return ::OCR_FindOneCoord(
        big_data, big_size, target_utf8, min_confidence, output, color_filter, 0, 0);
}

inline const char* YOLO_InferJson(
    int32_t handle, const uint8_t* big_data, int32_t big_size, float conf) {
    return ::YOLO_InferJson(handle, big_data, big_size, conf, 0, 0);
}
#endif
