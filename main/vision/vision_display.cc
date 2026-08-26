#include "vision_display.h"

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define TAG "VisionDisplay"

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

VisionDisplay::VisionDisplay() {
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

        if (style->draw_label) {
            char label_buf[64];
            if (!d.class_name.empty()) {
                snprintf(label_buf, sizeof(label_buf), "%s %.2f",
                         d.class_name.c_str(), d.confidence);
            } else {
                snprintf(label_buf, sizeof(label_buf), "#%d %.2f",
                         d.class_id, d.confidence);
            }
            int scale = (h >= 480) ? 4 : 3;
            int label_h = 8 * scale;
            int text_w = MeasureText5x7(label_buf, scale);
            int lx = b.x;
            int ly = b.y - label_h - 2;
            if (ly < 0) ly = b.y + b.height - label_h - 2;
            if (ly < 0) ly = 0;
            if (lx < 0) lx = 0;
            if (lx + text_w > w) lx = w - text_w;
            FillRect(buf, w, h, stride, lx, ly, text_w, label_h, style->label_bg_rgb565);
            DrawText5x7(buf, w, h, stride, lx, ly + scale, label_buf,
                        style->label_text_rgb565, style->label_bg_rgb565, scale);
        }
    }

    // 左上角叠加状态信息：情绪 / 坐姿 / 心率（放大字号，超出屏幕宽度时截断）
    int scale = (h >= 480) ? 5 : (h >= 320) ? 4 : 3;
    int info_y = 2;
    char line[96];
    const uint16_t kTextFg = 0xFFFF;   // 白
    const uint16_t kTextBg = 0x0000;   // 黑

    auto draw_info_line = [&](const char* text) {
        char tmp[96];
        size_t max_len = (size_t)((w - 4) / (6 * scale));
        snprintf(tmp, sizeof(tmp), "%s", text);
        if (strlen(tmp) > max_len) tmp[max_len] = '\0';
        int tw = MeasureText5x7(tmp, scale);
        FillRect(buf, w, h, stride, 2, info_y, tw, 8 * scale, kTextBg);
        DrawText5x7(buf, w, h, stride, 2, info_y + scale, tmp, kTextFg, kTextBg, scale);
        info_y += 10 * scale;
    };

    if (detections.emotion.available && !detections.emotion.label.empty()) {
        snprintf(line, sizeof(line), "EMO:%s %.2f", detections.emotion.label.c_str(),
                 detections.emotion.confidence);
        draw_info_line(line);
    }
    if (detections.posture.available && !detections.posture.state.empty()) {
        snprintf(line, sizeof(line), "POSE:%s", detections.posture.state.c_str());
        draw_info_line(line);
    }
    if (detections.heart_rate.available) {
        snprintf(line, sizeof(line), "HR:%.0fbpm", detections.heart_rate.bpm);
        draw_info_line(line);
    } else if (!detections.heart_rate.error_message.empty()) {
        snprintf(line, sizeof(line), "HR:%s", detections.heart_rate.error_message.c_str());
        draw_info_line(line);
    }
    if (detections.blood_pressure.available) {
        if (detections.blood_pressure.ready) {
            snprintf(line, sizeof(line), "BP EXP:%.0f/%.0f", detections.blood_pressure.sbp_mmHg,
                     detections.blood_pressure.dbp_mmHg);
        } else if (detections.blood_pressure.status == "collecting") {
            snprintf(line, sizeof(line), "BP:%.0f/%.0fs", detections.blood_pressure.duration_s,
                     detections.blood_pressure.required_window_s);
        } else {
            snprintf(line, sizeof(line), "BP:%s", detections.blood_pressure.status.c_str());
        }
        draw_info_line(line);
    }
}

std::unique_ptr<LvglImage> VisionDisplay::ComposePreview(const uint8_t* frame_rgb565,
                                                          int width, int height,
                                                          size_t stride_bytes,
                                                          const DetectionResult& detections) {
    if (frame_rgb565 == nullptr || width <= 0 || height <= 0) {
        ESP_LOGE(TAG, "Invalid frame for preview composition");
        return nullptr;
    }

    size_t dst_stride = (size_t)width * 2;
    size_t total = dst_stride * (size_t)height;
    uint8_t* dst = (uint8_t*)heap_caps_malloc(total, MALLOC_CAP_SPIRAM);
    if (dst == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes for preview", (int)total);
        return nullptr;
    }

    if (stride_bytes == dst_stride) {
        memcpy(dst, frame_rgb565, total);
    } else {
        for (int y = 0; y < height; y++) {
            memcpy(dst + (size_t)y * dst_stride,
                   frame_rgb565 + (size_t)y * stride_bytes,
                   dst_stride);
        }
    }

    // Swap RGB565 byte pairs for LVGL compatibility (LVGL expects BGR565 byte order on ESP32-S3)
    {
        uint16_t* pixels = (uint16_t*)dst;
        size_t pixel_count = total / 2;
        for (size_t i = 0; i < pixel_count; i++) {
            pixels[i] = __builtin_bswap16(pixels[i]);
        }
    }

    DrawDetections(dst, width, height, dst_stride, detections);

    return std::make_unique<LvglAllocatedImage>(dst, total, width, height, (int)dst_stride, LV_COLOR_FORMAT_RGB565);
}
