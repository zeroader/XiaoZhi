#ifndef VISION_DISPLAY_H
#define VISION_DISPLAY_H

#include "detector.h"
#include "lvgl_image.h"

#include <memory>
#include <string>
#include <cstdint>

struct VisionFrameBufferPool;

struct BoxStyle {
    uint16_t border_color_rgb565;
    uint16_t fill_color_rgb565;
    uint16_t label_bg_rgb565;
    uint16_t label_text_rgb565;
    int border_thickness;
    bool draw_label;

    BoxStyle();
};

class VisionDisplay {
public:
    VisionDisplay();
    ~VisionDisplay();

    void SetDefaultStyle(const BoxStyle& style);

    std::unique_ptr<LvglImage> ComposePreview(const uint8_t* frame_rgb565,
                                               int width, int height,
                                               size_t stride_bytes,
                                               const DetectionResult& detections,
                                               int target_width = 0,
                                               int target_height = 0);

    void DrawDetections(uint8_t* buf_rgb565,
                        int width, int height, size_t stride_bytes,
                        const DetectionResult& detections,
                        const BoxStyle* style_override = nullptr);

    static uint16_t Rgb888To565(uint8_t r, uint8_t g, uint8_t b);

private:
    BoxStyle default_style_;
    // Display-sized RGB565 preview buffers are kept alive by the LCD for
    // several frames. Recycle them once released instead of allocating from
    // PSRAM on every frame, which fragments memory during long sessions.
    std::shared_ptr<VisionFrameBufferPool> frame_buffer_pool_;

    void DrawRect(uint8_t* buf_rgb565, int w, int h, size_t stride,
                  int x, int y, int rw, int rh,
                  uint16_t color, int thickness);

    void DrawHLine(uint8_t* buf_rgb565, int w, int h, size_t stride,
                   int x, int y, int len, uint16_t color);

    void DrawVLine(uint8_t* buf_rgb565, int w, int h, size_t stride,
                   int x, int y, int len, uint16_t color);

    void FillRect(uint8_t* buf_rgb565, int w, int h, size_t stride,
                  int x, int y, int rw, int rh, uint16_t color);

    void DrawChar5x7(uint8_t* buf_rgb565, int w, int h, size_t stride,
                     int x, int y, char c, uint16_t fg, uint16_t bg,
                     int scale = 1);

    void DrawText5x7(uint8_t* buf_rgb565, int w, int h, size_t stride,
                     int x, int y, const char* text, uint16_t fg, uint16_t bg,
                     int scale = 1);

    int MeasureText5x7(const char* text, int scale = 1);
};

#endif
