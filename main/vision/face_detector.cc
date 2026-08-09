#include "face_detector.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <cstring>

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
#if __has_include("esp_face_detect.h")
#define HAVE_ESP_FACE_DETECT 1
#include "esp_face_detect.h"
#include "esp_face_detect_default_run_model.h"
#include "dl_image_draw.h"
#endif
#endif

#define TAG "FaceDetector"

FaceDetector::FaceDetector()
    : initialized_(false)
    , confidence_threshold_(0.5f)
    , sr_handle_(nullptr) {
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

#ifdef HAVE_ESP_FACE_DETECT
    ESP_LOGI(TAG, "Initializing ESP-SR face detector");
    face_detect_model_data_t *model = NULL;
    model = get_face_detect_model();
    if (model == NULL) {
        ESP_LOGE(TAG, "Failed to get face detect model");
        return false;
    }

    sr_handle_ = esp_face_detect_create(model);
    if (sr_handle_ == nullptr) {
        ESP_LOGE(TAG, "Failed to create face detector");
        return false;
    }

    esp_face_detect_set_threshold(sr_handle_, confidence_threshold_);
    initialized_ = true;
    ESP_LOGI(TAG, "ESP-SR face detector initialized, threshold=%.2f", confidence_threshold_);
#else
    ESP_LOGW(TAG, "ESP-SR face detection not available in this build. "
                  "Using stub detector - will return empty results. "
                  "For real face detection, enable esp-sr with face_detect model.");
    initialized_ = true;
#endif
    return initialized_;
}

void FaceDetector::Deinitialize() {
    if (!initialized_) {
        return;
    }

#ifdef HAVE_ESP_FACE_DETECT
    if (sr_handle_ != nullptr) {
        esp_face_detect_destroy((esp_face_detect_handle_t)sr_handle_);
        sr_handle_ = nullptr;
    }
#endif
    initialized_ = false;
    ESP_LOGI(TAG, "Face detector deinitialized");
}

DetectionResult FaceDetector::Detect(const ImageFrame& frame) {
    DetectionResult result;
    result.source_width = frame.width;
    result.source_height = frame.height;
    result.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000LL);

    if (!initialized_ || frame.data == nullptr || frame.width <= 0 || frame.height <= 0) {
        return result;
    }

#ifdef HAVE_ESP_FACE_DETECT
    auto start = esp_timer_get_time();

    esp_face_detect_result_list_t* faces = (esp_face_detect_result_list_t*)
        esp_face_detect_run((esp_face_detect_handle_t)sr_handle_,
                            (uint8_t*)frame.data, frame.width, frame.height,
                            (pix_format_t)frame.pixel_format);

    if (faces == nullptr) {
        ESP_LOGW(TAG, "Face detect returned null");
        return result;
    }

    int count = faces->count;
    for (int i = 0; i < count; i++) {
        auto& face = faces->results[i];
        if (face.score < confidence_threshold_) {
            continue;
        }
        Detection d;
        d.class_id = 0;
        d.class_name = "face";
        d.confidence = face.score;
        d.box = BoundingBox(face.box[0], face.box[1],
                            face.box[2] - face.box[0],
                            face.box[3] - face.box[1]);
        result.detections.push_back(d);
    }

    auto end = esp_timer_get_time();
    ESP_LOGD(TAG, "Detected %d faces in %d ms (threshold=%.2f)",
             (int)result.detections.size(),
             (int)((end - start) / 1000LL),
             confidence_threshold_);
#else
    (void)frame;
#endif

    return result;
}

void FaceDetector::SetConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
#ifdef HAVE_ESP_FACE_DETECT
    if (sr_handle_ != nullptr) {
        esp_face_detect_set_threshold((esp_face_detect_handle_t)sr_handle_, threshold);
    }
#endif
}

float FaceDetector::GetConfidenceThreshold() const {
    return confidence_threshold_;
}
