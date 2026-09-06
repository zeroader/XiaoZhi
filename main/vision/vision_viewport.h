#pragma once

#include <algorithm>
#include <cstdint>

// Shared by the dashboard, preview composer and HTTP recognition request.
struct VisionViewport {
    int x, y, width, height;
};

inline VisionViewport VisionVideoViewport(int screen_width, int screen_height) {
    const int side = std::max(64, screen_width * 70 / 320);
    const int top = std::max(20, screen_height * 28 / 240) + 4;
    return {side + 5, top, screen_width - side * 2 - 4,
            screen_height * 199 / 240 - top - 4};
}

inline VisionViewport VisionCenterCrop(int width, int height, VisionViewport viewport) {
    int crop_width = width, crop_height = height;
    if (int64_t(width) * viewport.height > int64_t(height) * viewport.width) {
        crop_width = std::max(1, int(int64_t(height) * viewport.width / viewport.height));
    } else {
        crop_height = std::max(1, int(int64_t(width) * viewport.height / viewport.width));
    }
    return {(width - crop_width) / 2, (height - crop_height) / 2, crop_width, crop_height};
}
