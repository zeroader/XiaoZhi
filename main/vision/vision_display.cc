#include "vision_display.h"

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>

#define TAG "VisionDisplay"

struct VisionFrameBufferPool {
    struct Buffer {
        uint8_t* data;
        size_t size;
    };

    ~VisionFrameBufferPool() {
        for (size_t i = 0; i < free_count; ++i) {
            heap_caps_free(free_buffers[i].data);
        }
    }

    uint8_t* Acquire(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t i = 0; i < free_count; ++i) {
            if (free_buffers[i].size == size) {
                uint8_t* data = free_buffers[i].data;
                free_buffers[i] = free_buffers[--free_count];
                return data;
            }
        }
        return static_cast<uint8_t*>(
            heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }

    void Release(uint8_t* data, size_t size) {
        if (data == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex);
        // In steady state one spare buffer is sufficient (composition happens
        // just before the LCD releases its oldest retained frame). Keep a
        // second slot for a temporary resolution transition.
        if (free_count < 2) {
            free_buffers[free_count++] = {data, size};
        } else {
            heap_caps_free(data);
        }
    }

    std::mutex mutex;
    Buffer free_buffers[2]{};
    size_t free_count = 0;
};

namespace {

class PooledVisionImage final : public LvglImage {
public:
    PooledVisionImage(std::shared_ptr<VisionFrameBufferPool> pool, uint8_t* data,
                      size_t size, int width, int height, int stride)
        : pool_(std::move(pool)), data_(data), size_(size) {
        image_dsc_.data_size = size;
        image_dsc_.data = data;
        image_dsc_.header.magic = LV_IMAGE_HEADER_MAGIC;
        image_dsc_.header.cf = LV_COLOR_FORMAT_RGB565;
        image_dsc_.header.w = width;
        image_dsc_.header.h = height;
        image_dsc_.header.stride = stride;
    }

    ~PooledVisionImage() override {
        pool_->Release(data_, size_);
        image_dsc_.data = nullptr;
    }

    const lv_img_dsc_t* image_dsc() const override { return &image_dsc_; }

private:
    std::shared_ptr<VisionFrameBufferPool> pool_;
    uint8_t* data_;
    size_t size_;
    lv_img_dsc_t image_dsc_{};
};

}  // namespace

static const uint8_t kFont5x7[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x5F,0x00,0x00}, {0x00,0x07,0x00,0x07,0x00},
    {0x14,0x7F,0x14,0x7F,0x14}, {0x24,0x2A,0x7F,0x2A,0x12}, {0x23,0x13,0x08,0x64,0x62},
    {0x36,0x49,0x55,0x22,0x50}, {0x00,0x05,0x03,0x00,0x00}, {0x00,0x1C,0x22,0x41,0x00},
    {0x00,0x41,0x22,0x1C,0x00}, {0x14,0x08,0x3E,0x08,0x14}, {0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x80,0x70,0x30,0x00}, {0x08,0x08,0x08,0x08,0x08}, {0x00,0x00,0x60,0x60,0x00},
    {0x20,0x10,0x08,0x04,0x02}, {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4B,0x31}, {0x18,0x14,0x12,0x7F,0x10},
    {0x27,0x45,0x45,0x45,0x39}, {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1E}, {0x00,0x36,0x36,0x00,0x00},
    {0x00,0x56,0x36,0x00,0x00}, {0x08,0x14,0x22,0x41,0x00}, {0x14,0x14,0x14,0x14,0x14},
    {0x00,0x41,0x22,0x14,0x08}, {0x02,0x01,0x51,0x09,0x06}, {0x32,0x49,0x59,0x51,0x3E},
    {0x7E,0x09,0x09,0x09,0x7E}, {0x7F,0x49,0x49,0x49,0x36}, {0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C}, {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
    {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F}, {0x00,0x41,0x7F,0x41,0x00},
    {0x20,0x40,0x41,0x3F,0x01}, {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40},
    {0x7F,0x02,0x0C,0x02,0x7F}, {0x7F,0x04,0x08,0x10,0x7F}, {0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06}, {0x3E,0x41,0x51,0x21,0x5E}, {0x7F,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01}, {0x3F,0x40,0x40,0x40,0x3F},
    {0x1F,0x20,0x40,0x20,0x1F}, {0x3F,0x40,0x38,0x40,0x3F}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}, {0x00,0x7F,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20}, {0x00,0x41,0x41,0x7F,0x00}, {0x04,0x02,0x01,0x02,0x04},
    {0x80,0x80,0x80,0x80,0x80}, {0x00,0x03,0x07,0x00,0x00}, {0x20,0x54,0x54,0x54,0x78},
    {0x7F,0x48,0x44,0x44,0x38}, {0x38,0x44,0x44,0x44,0x20}, {0x38,0x44,0x44,0x48,0x7F},
    {0x38,0x54,0x54,0x54,0x18}, {0x08,0x7E,0x09,0x01,0x02}, {0x18,0xA4,0xA4,0xA4,0x7C},
    {0x7F,0x08,0x04,0x04,0x78}, {0x00,0x44,0x7D,0x40,0x00}, {0x40,0x80,0x44,0x7D,0x00},
    {0x7F,0x10,0x28,0x44,0x00}, {0x00,0x41,0x7F,0x40,0x00}, {0x7C,0x04,0x78,0x04,0x78},
    {0x7C,0x08,0x04,0x04,0x78}, {0x38,0x44,0x44,0x44,0x38}, {0xFC,0x18,0x24,0x24,0x18},
    {0x18,0x24,0x24,0x18,0xFC}, {0x7C,0x08,0x04,0x04,0x08}, {0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20}, {0x3C,0x40,0x40,0x20,0x7C}, {0x1C,0x20,0x40,0x20,0x1C},
    {0x3C,0x40,0x30,0x40,0x3C}, {0x44,0x28,0x10,0x28,0x44}, {0x1C,0xA0,0xA0,0xA0,0x7C},
    {0x44,0x64,0x54,0x4C,0x44}, {0x00,0x08,0x36,0x41,0x00}, {0x00,0x00,0x77,0x00,0x00},
    {0x00,0x41,0x36,0x08,0x00}, {0x08,0x08,0x2A,0x1C,0x08}, {0x08,0x1C,0x2A,0x08,0x08},
    {0x1E,0x10,0x10,0x10,0x10}, {0x08,0x14,0x08,0x08,0x14}, {0x00,0x00,0x18,0x18,0x00},
};

BoxStyle::BoxStyle()
    : border_color_rgb565(0xF800)
    , fill_color_rgb565(0)
    , label_bg_rgb565(0xF800)
    , label_text_rgb565(0xFFFF)
    , border_thickness(2)
    , draw_label(true) {
}

VisionDisplay::VisionDisplay()
    : frame_buffer_pool_(std::make_shared<VisionFrameBufferPool>()) {
}

VisionDisplay::~VisionDisplay() {
}

void VisionDisplay::SetDefaultStyle(const BoxStyle& style) {
    default_style_ = style;
}

uint16_t VisionDisplay::Rgb888To565(uint8_t r, uint8_t g, uint8_t b) {
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

void VisionDisplay::DrawHLine(uint8_t* buf, int w, int h, size_t stride,
                              int x, int y, int len, uint16_t color) {
    if (y < 0 || y >= h) return;
    if (x + len <= 0 || x >= w) return;
    if (x < 0) { len += x; x = 0; }
    if (x + len > w) len = w - x;
    if (len <= 0) return;
    uint16_t* row = (uint16_t*)(buf + (size_t)y * stride);
    uint16_t c = __builtin_bswap16(color);
    for (int i = 0; i < len; i++) {
        row[x + i] = c;
    }
}

void VisionDisplay::DrawVLine(uint8_t* buf, int w, int h, size_t stride,
                              int x, int y, int len, uint16_t color) {
    if (x < 0 || x >= w) return;
    if (y + len <= 0 || y >= h) return;
    if (y < 0) { len += y; y = 0; }
    if (y + len > h) len = h - y;
    if (len <= 0) return;
    uint16_t c = __builtin_bswap16(color);
    for (int i = 0; i < len; i++) {
        uint16_t* row = (uint16_t*)(buf + (size_t)(y + i) * stride);
        row[x] = c;
    }
}

void VisionDisplay::DrawRect(uint8_t* buf, int w, int h, size_t stride,
                             int x, int y, int rw, int rh,
                             uint16_t color, int thickness) {
    for (int t = 0; t < thickness; t++) {
        DrawHLine(buf, w, h, stride, x,         y + t,       rw, color);
        DrawHLine(buf, w, h, stride, x,         y + rh - 1 - t, rw, color);
        DrawVLine(buf, w, h, stride, x + t,     y,            rh, color);
        DrawVLine(buf, w, h, stride, x + rw - 1 - t, y,        rh, color);
    }
}

void VisionDisplay::FillRect(uint8_t* buf, int w, int h, size_t stride,
                             int x, int y, int rw, int rh, uint16_t color) {
    if (x >= w || y >= h || rw <= 0 || rh <= 0) return;
    if (x < 0) { rw += x; x = 0; }
    if (y < 0) { rh += y; y = 0; }
    if (x + rw > w) rw = w - x;
    if (y + rh > h) rh = h - y;
    uint16_t c = __builtin_bswap16(color);
    for (int j = 0; j < rh; j++) {
        uint16_t* row = (uint16_t*)(buf + (size_t)(y + j) * stride);
        for (int i = 0; i < rw; i++) {
            row[x + i] = c;
        }
    }
}

void VisionDisplay::DrawChar5x7(uint8_t* buf, int w, int h, size_t stride,
                                int x, int y, char c, uint16_t fg, uint16_t bg,
                                int scale) {
    uint8_t idx = (uint8_t)c;
    // kFont5x7 表按 (ASCII-32) 索引（index 0=空格，16='0'），越界字符画成空格
    if (idx < 32 || idx > 127) idx = 32;
    idx -= 32;
    const uint8_t* glyph = kFont5x7[idx];
    for (int gy = 0; gy < 7; gy++) {
        for (int gx = 0; gx < 5; gx++) {
            bool on = (glyph[gx] & (1 << gy)) != 0;
            uint16_t color = on ? fg : bg;
            if (scale == 1) {
                int px = x + gx;
                int py = y + gy;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    uint16_t* row = (uint16_t*)(buf + (size_t)py * stride);
                    row[px] = __builtin_bswap16(color);
                }
            } else {
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + gx * scale + sx;
                        int py = y + gy * scale + sy;
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            uint16_t* row = (uint16_t*)(buf + (size_t)py * stride);
                            row[px] = __builtin_bswap16(color);
                        }
                    }
                }
            }
        }
    }
}

int VisionDisplay::MeasureText5x7(const char* text, int scale) {
    if (text == nullptr) return 0;
    int n = (int)strlen(text);
    return n * 6 * scale;
}

void VisionDisplay::DrawText5x7(uint8_t* buf, int w, int h, size_t stride,
                                int x, int y, const char* text, uint16_t fg, uint16_t bg,
                                int scale) {
    if (text == nullptr) return;
    int cx = x;
    for (const char* p = text; *p != '\0'; p++) {
        DrawChar5x7(buf, w, h, stride, cx, y, *p, fg, bg, scale);
        cx += 6 * scale;
    }
}

void VisionDisplay::DrawDetections(uint8_t* buf, int w, int h, size_t stride,
                                   const DetectionResult& detections,
                                   const BoxStyle* style_override) {
    const BoxStyle* style = style_override != nullptr ? style_override : &default_style_;

    for (const auto& d : detections.detections) {
        const auto& b = d.box;
        DrawRect(buf, w, h, stride, b.x, b.y, b.width, b.height,
                 style->border_color_rgb565, style->border_thickness);

    }
}

std::unique_ptr<LvglImage> VisionDisplay::ComposePreview(const uint8_t* frame_rgb565,
                                                          int width, int height,
                                                          size_t stride_bytes,
                                                          const DetectionResult& detections,
                                                          int target_width,
                                                          int target_height) {
    if (frame_rgb565 == nullptr || width <= 0 || height <= 0 || stride_bytes < size_t(width) * 2) {
        ESP_LOGE(TAG, "Invalid frame for preview composition");
        return nullptr;
    }

    if (target_width <= 0) target_width = width;
    if (target_height <= 0) target_height = height;

    // Crop to exactly the on-screen video's aspect ratio and downscale before
    // handing the frame to LVGL. Previously LVGL repeatedly scaled the full
    // 320x240 image during every partial flush, limiting the live view to only
    // a few FPS. Nearest-neighbour sampling is intentional here: it is fast,
    // deterministic, and the small LCD preview does not benefit from a costly
    // filtered resize.
    int crop_width = width;
    int crop_height = height;
    if (int64_t(width) * target_height > int64_t(height) * target_width) {
        crop_width = std::max(1, int(int64_t(height) * target_width / target_height));
    } else {
        crop_height = std::max(1, int(int64_t(width) * target_height / target_width));
    }
    const int crop_x = (width - crop_width) / 2;
    const int crop_y = (height - crop_height) / 2;

    const size_t dst_stride = size_t(target_width) * 2;
    const size_t total = dst_stride * size_t(target_height);
    uint8_t* dst = frame_buffer_pool_->Acquire(total);
    if (dst == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes for preview", (int)total);
        return nullptr;
    }

    const uint32_t x_step = uint32_t((uint64_t(crop_width) << 16) / target_width);
    const uint32_t y_step = uint32_t((uint64_t(crop_height) << 16) / target_height);
    uint32_t y_acc = y_step / 2;
    for (int y = 0; y < target_height; ++y, y_acc += y_step) {
        const int source_y = crop_y + std::min(crop_height - 1, int(y_acc >> 16));
        const uint8_t* source_row = frame_rgb565 + size_t(source_y) * stride_bytes;
        uint8_t* target_row = dst + size_t(y) * dst_stride;
        uint32_t x_acc = x_step / 2;
        for (int x = 0; x < target_width; ++x, x_acc += x_step) {
            const int source_x = crop_x + std::min(crop_width - 1, int(x_acc >> 16));
            const uint8_t* source = source_row + size_t(source_x) * 2;
            // Preserve the byte order expected by the existing LVGL/SPI path.
            target_row[size_t(x) * 2] = source[1];
            target_row[size_t(x) * 2 + 1] = source[0];
        }
    }

    const int source_width = detections.source_width > 0 ? detections.source_width : width;
    const int source_height = detections.source_height > 0 ? detections.source_height : height;
    for (const auto& detection : detections.detections) {
        const auto& box = detection.box;
        // Server responses are already mapped back to full-camera coordinates.
        const int source_x1 = box.x * width / source_width;
        const int source_y1 = box.y * height / source_height;
        const int source_x2 = (box.x + box.width) * width / source_width;
        const int source_y2 = (box.y + box.height) * height / source_height;
        const int clipped_x1 = std::max(crop_x, source_x1);
        const int clipped_y1 = std::max(crop_y, source_y1);
        const int clipped_x2 = std::min(crop_x + crop_width, source_x2);
        const int clipped_y2 = std::min(crop_y + crop_height, source_y2);
        if (clipped_x2 <= clipped_x1 || clipped_y2 <= clipped_y1) continue;
        const int box_x = (clipped_x1 - crop_x) * target_width / crop_width;
        const int box_y = (clipped_y1 - crop_y) * target_height / crop_height;
        const int box_x2 = (clipped_x2 - crop_x) * target_width / crop_width;
        const int box_y2 = (clipped_y2 - crop_y) * target_height / crop_height;
        DrawRect(dst, target_width, target_height, dst_stride,
                 box_x, box_y, std::max(1, box_x2 - box_x), std::max(1, box_y2 - box_y),
                 default_style_.border_color_rgb565, default_style_.border_thickness);
    }

    auto* pooled_image = new (std::nothrow) PooledVisionImage(
        frame_buffer_pool_, dst, total, target_width, target_height,
        static_cast<int>(dst_stride));
    if (pooled_image == nullptr) {
        frame_buffer_pool_->Release(dst, total);
        ESP_LOGE(TAG, "Failed to allocate preview image descriptor");
        return nullptr;
    }
    return std::unique_ptr<LvglImage>(pooled_image);
}
