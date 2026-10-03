#include "yolo_preprocess.h"
#include "bmp_view.h"
#include <cassert>
#include <cstring>
#include <random>
#include <iostream>
using ai::channels_for_format;
namespace reference {
struct LetterboxInfo {
    float scale;   // scaling factor applied to the original image
    float pad_x;   // horizontal padding (pixels in resized coordinate)
    float pad_y;   // vertical padding (pixels in resized coordinate)
};

// letterbox 缩放：保持宽高比把 src 缩放到 dst_w x dst_h 内，并用灰色 114 填充边缘。
LetterboxInfo letterbox_resize(const uint8_t* src, int src_w, int src_h, int src_stride, int channels,
                               std::vector<uint8_t>& dst, int dst_w, int dst_h) {
    LetterboxInfo info{};
    const float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
    info.scale = std::min(scale_w, scale_h);

    const int new_w = static_cast<int>(std::round(src_w * info.scale));
    const int new_h = static_cast<int>(std::round(src_h * info.scale));
    info.pad_x = (dst_w - new_w) / 2.0f;
    info.pad_y = (dst_h - new_h) / 2.0f;

    const int pad_left = static_cast<int>(info.pad_x);
    const int pad_top = static_cast<int>(info.pad_y);

    dst.assign(static_cast<size_t>(dst_w) * dst_h * 3, 114);

    // 使用最近邻缩放到居中区域，对 YOLO 输入已经足够。
    for (int dy = 0; dy < new_h; ++dy) {
        const float src_y = dy / info.scale;
        const int sy = std::min(static_cast<int>(src_y), src_h - 1);
        uint8_t* dst_row = dst.data() + (static_cast<size_t>(dy + pad_top) * dst_w + pad_left) * 3;
        const uint8_t* src_row = src + static_cast<std::ptrdiff_t>(sy) * static_cast<std::ptrdiff_t>(src_stride);
        for (int dx = 0; dx < new_w; ++dx) {
            const float src_x = dx / info.scale;
            const int sx = std::min(static_cast<int>(src_x), src_w - 1);
            const uint8_t* sp = src_row + sx * channels;
            uint8_t* dp = dst_row + dx * 3;
            // 复制像素；此处保留调用方提供的原通道顺序。
            dp[0] = sp[0];
            dp[1] = (channels >= 3) ? sp[1] : sp[0];
            dp[2] = (channels >= 3) ? sp[2] : sp[0];
        }
    }

    return info;
}

// 将 AIImage 转为 RGB 顺序、归一化到 [0,1] 的 float32 CHW 张量。
// 支持 BGR24/RGB24/GRAY8/BGRA32/RGBA32 输入格式。
void image_to_chw_float(
    const uint8_t* data,
    int width,
    int height,
    int stride,
    int format,
    int target_w,
    int target_h,
    std::vector<float>& output,
    std::vector<uint8_t>& resized,
    LetterboxInfo& info) {
    const int channels = channels_for_format(format);

    // 第 1 步：letterbox 缩放，输出仍按源通道顺序，三通道。
    info = letterbox_resize(data, width, height, stride, channels, resized, target_w, target_h);

    // 第 2 步：HWC 转 CHW float，并归一化、重排为 RGB。
    const size_t pixels = static_cast<size_t>(target_w) * target_h;
    output.resize(3 * pixels);
    float* r_plane = output.data();
    float* g_plane = output.data() + pixels;
    float* b_plane = output.data() + 2 * pixels;

    const bool is_bgr = (format == AI_IMAGE_BGR24 || format == AI_IMAGE_BGRA32);
    // GRAY 会在 letterbox_resize 中扩展为 3 个相同通道。

    for (size_t i = 0; i < pixels; ++i) {
        const uint8_t* p = resized.data() + i * 3;
        if (is_bgr) {
            // 源顺序为 B G R，需要写入 R G B 平面。
            r_plane[i] = p[2] / 255.0f;
            g_plane[i] = p[1] / 255.0f;
            b_plane[i] = p[0] / 255.0f;
        } else {
            // 源顺序为 R G B，或由 GRAY 扩展而来。
            r_plane[i] = p[0] / 255.0f;
            g_plane[i] = p[1] / 255.0f;
            b_plane[i] = p[2] / 255.0f;
        }
    }
}
}
int main() {
    std::mt19937 random(42);
    for (int format : {AI_IMAGE_GRAY8, AI_IMAGE_RGB24, AI_IMAGE_BGR24, AI_IMAGE_BGRA32, AI_IMAGE_RGBA32}) {
        for (const auto dims : {std::pair<int,int>{800,600}, {321,205}, {17,19}, {1,320}, {640,1}}) {
            const int stride = dims.first * ai::channels_for_format(format) + 7;
            std::vector<uint8_t> pixels(static_cast<size_t>(stride) * dims.second);
            for (auto& p : pixels) p = static_cast<uint8_t>(random());
            for (int direction : {1,-1}) {
                AIImage image{pixels.data() + (direction == 1 ? 0 : static_cast<size_t>(stride) * (dims.second-1)), dims.first,dims.second,direction*stride,format};
                std::vector<float> expected, actual(3*320*320);
                std::vector<uint8_t> scratch; std::vector<int> columns;
                reference::LetterboxInfo previous{};
                reference::image_to_chw_float(image.data,image.width,image.height,image.stride,image.format,320,320,expected,scratch,previous);
                const auto box=ai::yolo_preprocess(image,320,320,actual.data(),columns);
                assert(previous.scale == box.scale && previous.pad_x == box.pad_x && previous.pad_y == box.pad_y);
                assert(expected.size() == actual.size());
                assert(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)) == 0);
            }
        }
    }
    for (int direction : {1,-1}) {
        std::vector<uint8_t> bmp(54 + 8*3, 0);
        bmp[0]='B';bmp[1]='M';
        const auto set=[&](int offset,int32_t value) { std::memcpy(bmp.data()+offset,&value,4); };
        set(10,54);set(14,40);set(18,2);set(22,direction*3);bmp[26]=1;bmp[28]=24;
        AIImage view{};assert(ai::bmp24_view(bmp.data(),bmp.size(),&view));
        assert(view.width==2 && view.height==3 && view.stride==-direction*8);
        assert(!ai::bmp24_view(bmp.data(),bmp.size()-1,&view));
        set(22,INT32_MIN);assert(!ai::bmp24_view(bmp.data(),bmp.size(),&view));
        set(22,3);set(10,UINT32_MAX);assert(!ai::bmp24_view(bmp.data(),bmp.size(),&view));
    }
    std::cout << "fused preprocess bit-identical, signed BMP stride and overflow checks passed\n";
}
