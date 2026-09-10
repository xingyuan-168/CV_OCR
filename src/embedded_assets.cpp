#include "embedded_assets.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ai {

#if defined(_WIN32)
namespace {

// 获取当前 ai_engine.dll 的模块句柄，避免调用方工作目录影响资源读取。
HMODULE current_module() {
    HMODULE module = nullptr;
    const auto address = reinterpret_cast<LPCWSTR>(&current_module);
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address,
            &module)) {
        return nullptr;
    }
    return module;
}

} // namespace
#endif

// 从 Windows RCDATA 资源中取出模型/字典字节。
bool get_embedded_asset(EmbeddedAssetId id, EmbeddedAsset* output) {
    if (output == nullptr) {
        return false;
    }
    output->data = nullptr;
    output->size = 0;

#if defined(_WIN32) && defined(AIENGINE_EMBED_ASSETS)
    HMODULE module = current_module();
    if (module == nullptr) {
        return false;
    }

    HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(static_cast<int>(id)), MAKEINTRESOURCEW(10));
    if (resource == nullptr) {
        return false;
    }

    HGLOBAL loaded = LoadResource(module, resource);
    if (loaded == nullptr) {
        return false;
    }

    const DWORD size = SizeofResource(module, resource);
    const void* data = LockResource(loaded);
    if (data == nullptr || size == 0) {
        return false;
    }

    output->data = data;
    output->size = static_cast<size_t>(size);
    return true;
#else
    (void)id;
    return false;
#endif
}

// 确认四个运行必需资源都存在。
bool has_embedded_assets() {
    EmbeddedAsset asset;
    return get_embedded_asset(EmbeddedAssetId::YoloModel, &asset) &&
        get_embedded_asset(EmbeddedAssetId::YoloLabels, &asset) &&
        get_embedded_asset(EmbeddedAssetId::OcrRecModel, &asset) &&
        get_embedded_asset(EmbeddedAssetId::OcrCharset, &asset);
}

} // namespace ai
