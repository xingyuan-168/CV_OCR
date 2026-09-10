#pragma once

#include <cstddef>

namespace ai {

// DLL 内置资源编号。编号值需要与 embedded_assets.rc.in 保持一致。
enum class EmbeddedAssetId {
    YoloModel = 101,
    YoloLabels = 102,
    OcrDetModel = 103,
    OcrRecModel = 104,
    OcrCharset = 105
};

// 指向 DLL 资源中的只读字节视图；生命周期由模块资源管理。
struct EmbeddedAsset {
    const void* data = nullptr;
    size_t size = 0;
};

// 从当前 DLL 模块读取指定内置资源。
bool get_embedded_asset(EmbeddedAssetId id, EmbeddedAsset* output);

// 判断当前 DLL 是否编译了完整的默认模型资源。
bool has_embedded_assets();

} // namespace ai
