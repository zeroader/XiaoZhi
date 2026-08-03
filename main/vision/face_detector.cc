#include "face_detector.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cstdlib>

#include <human_face_detect.hpp>
#include <dl_image_define.hpp>

#define TAG "FaceDetector"

FaceDetector::FaceDetector()
    : initialized_(false)
    , confidence_threshold_(0.5f)
    , detector_(nullptr) {
}

FaceDetector::~FaceDetector() {
    Deinitialize();
}

const char* FaceDetector::GetName() const {
    return "face";
}

bool FaceDetector::Initialize() {
    if (initialized_) {
        return true;
    }

    detector_ = new HumanFaceDetect();
    if (detector_ == nullptr) {
        ESP_LOGE(TAG, "Failed to create HumanFaceDetect instance");
        return false;
    }

    initialized_ = true;
    ESP_LOGI(TAG, "ESP-DL face detector initialized (MSR_S8_V1+MNP_S8_V1, input=%dx%d, threshold=%.2f)",
             kModelWidth, kModelHeight, confidence_threshold_);
    return true;
}

void FaceDetector::Deinitialize() {
    if (!initialized_) {
        return;
    }

    if (detector_ != nullptr) {
        delete detector_;
        detector_ = nullptr;
    }
    initialized_ = false;
    ESP_LOGI(TAG, "Face detector deinitialized");
}

void FaceDetector::Rgb565ToRgb888(const uint8_t* src, uint8_t* dst, int width, int height) {
    const uint16_t* src16 = (const uint16_t*)src;
    int pixel_count = width * height;
    for (int i = 0; i < pixel_count; i++) {
        uint16_t pixel = src16[i];
        uint8_t r5 = (pixel >> 11) & 0x1F;
        uint8_t g6 = (pixel >> 5) & 0x3F;
        uint8_t b5 = pixel & 0x1F;
        // Scale to 8-bit per channel
        dst[i * 3 + 0] = (r5 << 3) | (r5 >> 2);   // R
        dst[i * 3 + 1] = (g6 << 2) | (g6 >> 4);   // G
        dst[i * 3 + 2] = (b5 << 3) | (b5 >> 2);   // B
    }
}

void FaceDetector::ResizeRgb888(const uint8_t* src, int src_w, int src_h,
                                uint8_t* dst, int dst_w, int dst_h) {
    for (int y = 0; y < dst_h; y++) {
        int src_y = y * src_h / dst_h;
        for (int x = 0; x < dst_w; x++) {
            int src_x = x * src_w / dst_w;
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx = (y * dst_w + x) * 3;
            dst[dst_idx + 0] = src[src_idx + 0];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }
}

DetectionResult FaceDetector::Detect(const ImageFrame& frame) {
    DetectionResult result;
    result.source_width = frame.width;
    result.source_height = frame.height;
    result.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000LL);

    if (!initialized_ || detector_ == nullptr
        || frame.data == nullptr || frame.width <= 0 || frame.height <= 0) {
        return result;
    }

    auto t_start = esp_timer_get_time();

    // Copy camera RGB565 frame to a separate PSRAM buffer to isolate from DMA.
    // HumanFaceDetect::run() internally handles: RGB565→RGB888 conversion,
    // resize to model input, normalization, quantization, anchor decode, and NMS.
    int frame_bytes = frame.width * frame.height * 2;
    uint8_t* frame_copy = (uint8_t*)heap_caps_malloc(frame_bytes, MALLOC_CAP_SPIRAM);
    if (frame_copy == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate frame copy (%d bytes)", frame_bytes);
        return result;
    }
    memcpy(frame_copy, frame.data, frame_bytes);

    dl::image::img_t img = {
        .data = frame_copy,
        .width = (uint16_t)frame.width,
        .height = (uint16_t)frame.height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE
    };

    auto t_infer_start = esp_timer_get_time();
    auto& detect_results = detector_->run(img);
    auto t_infer_end = esp_timer_get_time();

    // Coordinates are in original image space (model handles decode + clamp internally)
    for (const auto& res : detect_results) {
        ESP_LOGI(TAG, "Raw result: score=%.3f box=[%d,%d,%d,%d] keypoints=%d",
                 res.score, res.box[0], res.box[1], res.box[2], res.box[3],
                 (int)res.keypoint.size());
        if (res.score < confidence_threshold_) {
            continue;
        }
        // Filter edge-touching false positives (camera sensor artifacts)
        if (res.box[0] < 8 || res.box[1] < 8
            || res.box[2] > frame.width - 8
            || res.box[3] > frame.height - 8) {
            continue;
        }
        Detection d;
        d.class_id = 0;
        d.class_name = "face";
        d.confidence = res.score;
        int x1 = res.box[0];
        int y1 = res.box[1];
        int x2 = res.box[2];
        int y2 = res.box[3];
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > frame.width) x2 = frame.width;
        if (y2 > frame.height) y2 = frame.height;
        d.box = BoundingBox(x1, y1, x2 - x1, y2 - y1);
        result.detections.push_back(d);
    }

    heap_caps_free(frame_copy);

    auto t_end = esp_timer_get_time();
    ESP_LOGD(TAG, "Detected %d faces: total=%dms (infer=%dms)",
             (int)result.detections.size(),
             (int)((t_end - t_start) / 1000LL),
             (int)((t_infer_end - t_infer_start) / 1000LL));

    return result;
}

void FaceDetector::SetConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
    // HumanFaceDetect doesn't have a runtime threshold setter;
    // filtering is done in Detect() against confidence_threshold_
    ESP_LOGI(TAG, "Confidence threshold set to %.2f", threshold);
}

float FaceDetector::GetConfidenceThreshold() const {
    return confidence_threshold_;
}
