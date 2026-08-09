#include "online_detector.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cJSON.h>

#include "board.h"
#include "system_info.h"
#include "settings.h"
#include "image_to_jpeg.h"

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#define TAG "OnlineDetector"

static const char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

// JpegChunk for streaming JPEG encode -> PSRAM buffer
struct JpegChunk {
    uint8_t* data;
    size_t len;
};

OnlineDetector::OnlineDetector()
    : initialized_(false)
    , timeout_sec_(10)
    , jpeg_quality_(75)
    , task_(kTaskFaceEmotion)
    , frame_id_(0)
    , auto_frame_id_(true) {
}

OnlineDetector::~OnlineDetector() {
    Deinitialize();
}

const char* OnlineDetector::GetName() const {
    return "online";
}

bool OnlineDetector::Initialize() {
    if (initialized_) return true;

    // Load persisted configuration from NVS (saved by SetEndpoint/SetTimeoutSec/SetJpegQuality)
    {
        Settings settings("vision");
        std::string url = settings.GetString("online_url");
        if (!url.empty()) {
            endpoint_url_ = url;
        }
        int32_t timeout = settings.GetInt("online_timeout", 0);
        if (timeout > 0) {
            timeout_sec_ = timeout;
        }
        int32_t quality = settings.GetInt("online_quality", 0);
        if (quality >= 10 && quality <= 100) {
            jpeg_quality_ = quality;
        }
        std::string task = settings.GetString("online_task");
        if (task == kTaskPosture || task == kTaskHeartRate || task == kTaskFaceEmotion) {
            task_ = task;
        }
    }

    if (endpoint_url_.empty()) {
        ESP_LOGW(TAG, "Endpoint URL not set");
    }
    initialized_ = true;
    ESP_LOGI(TAG, "Online detector initialized, endpoint=%s, timeout=%ds, jpeg_quality=%d, task=%s",
             endpoint_url_.empty() ? "(not set)" : endpoint_url_.c_str(),
             timeout_sec_, jpeg_quality_, task_.c_str());
    return true;
}

void OnlineDetector::Deinitialize() {
    if (!initialized_) return;
    initialized_ = false;
    ESP_LOGI(TAG, "Online detector deinitialized");
}

std::string OnlineDetector::Base64Encode(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i + 1 < len) n |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) n |= (uint32_t)data[i + 2];
        result.push_back(kBase64Table[(n >> 18) & 0x3F]);
        result.push_back(kBase64Table[(n >> 12) & 0x3F]);
        result.push_back((i + 1 < len) ? kBase64Table[(n >> 6) & 0x3F] : '=');
        result.push_back((i + 2 < len) ? kBase64Table[n & 0x3F] : '=');
    }
    return result;
}

std::string OnlineDetector::BuildRequestBody(const std::string& base64_image, int width, int height) {
    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "frame_id", (double)frame_id_);
    cJSON_AddStringToObject(root, "task", task_.c_str());
    cJSON* image = cJSON_AddObjectToObject(root, "image");
    cJSON_AddStringToObject(image, "data", base64_image.c_str());
    cJSON_AddNumberToObject(image, "width", width);
    cJSON_AddNumberToObject(image, "height", height);
    cJSON_AddStringToObject(image, "format", "jpeg");

    char* json_str = cJSON_PrintUnformatted(root);
    std::string result(json_str);
    cJSON_free(json_str);
    cJSON_Delete(root);
    return result;
}

DetectionResult OnlineDetector::Detect(const ImageFrame& frame) {
    DetectionResult result;
    result.source_width = frame.width;
    result.source_height = frame.height;
    result.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000LL);
    result.task = task_;
    if (auto_frame_id_) {
        frame_id_++;
    }
    result.frame_id = frame_id_;

    if (!initialized_) {
        ESP_LOGE(TAG, "Not initialized");
        result.connection_ok = false;
        result.error_message = "detector not initialized";
        return result;
    }
    if (endpoint_url_.empty()) {
        ESP_LOGE(TAG, "Endpoint URL not set");
        result.connection_ok = false;
        result.error_message = "server URL not configured";
        return result;
    }
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) {
        ESP_LOGE(TAG, "Invalid frame");
        result.connection_ok = false;
        result.error_message = "invalid camera frame";
        return result;
    }

    auto total_start = esp_timer_get_time();

    // Step 1: JPEG encode the RGB565 frame
    QueueHandle_t jpeg_queue = xQueueCreate(40, sizeof(JpegChunk));
    if (jpeg_queue == nullptr) {
        ESP_LOGE(TAG, "Failed to create JPEG queue");
        return result;
    }

    std::thread encoder_thread([&frame, jpeg_queue, this]() {
        image_to_jpeg_cb((uint8_t*)frame.data, frame.len, frame.width, frame.height,
                         (pixformat_t)frame.pixel_format, jpeg_quality_,
            [](void* arg, size_t /*index*/, const void* data, size_t len) -> size_t {
                auto queue = (QueueHandle_t)arg;
                JpegChunk chunk = {
                    .data = (uint8_t*)heap_caps_aligned_alloc(16, len, MALLOC_CAP_SPIRAM),
                    .len = len
                };
                if (chunk.data != nullptr) {
                    memcpy(chunk.data, data, len);
                    xQueueSend(queue, &chunk, portMAX_DELAY);
                }
                return len;
            }, jpeg_queue);
        JpegChunk sentinel = { .data = nullptr, .len = 0 };
        xQueueSend(jpeg_queue, &sentinel, portMAX_DELAY);
    });

    // Collect all JPEG chunks into one buffer
    size_t jpeg_total = 0;
    uint8_t* jpeg_buffer = nullptr;

    while (true) {
        JpegChunk chunk;
        if (xQueueReceive(jpeg_queue, &chunk, pdMS_TO_TICKS(5000)) != pdPASS) {
            ESP_LOGE(TAG, "Timeout receiving JPEG chunk");
            break;
        }
        if (chunk.data == nullptr) break;  // sentinel

        uint8_t* new_buf = (uint8_t*)heap_caps_realloc(jpeg_buffer, jpeg_total + chunk.len,
                                                        MALLOC_CAP_SPIRAM);
        if (new_buf == nullptr) {
            heap_caps_free(chunk.data);
            ESP_LOGE(TAG, "Failed to grow JPEG buffer");
            if (jpeg_buffer) heap_caps_free(jpeg_buffer);
            jpeg_buffer = nullptr;
            break;
        }
        jpeg_buffer = new_buf;
        memcpy(jpeg_buffer + jpeg_total, chunk.data, chunk.len);
        jpeg_total += chunk.len;
        heap_caps_free(chunk.data);
    }
    encoder_thread.join();
    vQueueDelete(jpeg_queue);

    if (jpeg_buffer == nullptr || jpeg_total == 0) {
        ESP_LOGE(TAG, "JPEG encode failed");
        return result;
    }

    size_t jpeg_size = jpeg_total;

    // Step 2: Base64 encode the JPEG
    std::string base64_image = Base64Encode(jpeg_buffer, jpeg_size);
    heap_caps_free(jpeg_buffer);

    // Step 3: Build JSON request body
    std::string request_body = BuildRequestBody(base64_image, frame.width, frame.height);

    auto t_encode = esp_timer_get_time();
    ESP_LOGI(TAG, "Encoded: jpeg=%d B -> base64=%d B -> json=%d B (encode=%dms)",
             (int)jpeg_size,
             (int)base64_image.size(),
             (int)request_body.size(),
             (int)((t_encode - total_start) / 1000LL));

    // Step 4: HTTP POST
    auto network = Board::GetInstance().GetNetwork();
    auto http = network->CreateHttp(timeout_sec_);

    http->SetHeader("Content-Type", "application/json");
    http->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());

    http->SetContent(std::string(request_body));

    if (!http->Open("POST", endpoint_url_)) {
        ESP_LOGE(TAG, "Failed to open HTTP connection to %s", endpoint_url_.c_str());
        result.connection_ok = false;
        result.error_message = "cannot connect to detection server";
        http->Close();
        return result;
    }

    if (http->GetStatusCode() != 200) {
        ESP_LOGE(TAG, "HTTP status=%d", http->GetStatusCode());
        result.connection_ok = false;
        result.error_message = "detection server returned error " + std::to_string(http->GetStatusCode());
        http->Close();
        return result;
    }

    std::string response = http->ReadAll();
    http->Close();

    auto t_http = esp_timer_get_time();

    // Step 5: Parse response
    ParseResponse(response, result);

    auto total_end = esp_timer_get_time();
    ESP_LOGI(TAG, "Detect: task=%s frame=%d jpeg=%d B, detections=%d, total=%dms (encode=%dms, http=%dms) server(dec=%.1fms,inf=%.1fms,total=%.1fms)",
             task_.c_str(),
             (int)frame_id_,
             (int)jpeg_size,
             (int)result.detections.size(),
             (int)((total_end - total_start) / 1000LL),
             (int)((t_encode - total_start) / 1000LL),
             (int)((t_http - t_encode) / 1000LL),
             result.decode_ms, result.infer_ms, result.total_ms);

    return result;
}

// 解析 bbox 对象 {x,y,width,height} 或 {x1,y1,x2,y2}，成功返回 true
static bool ParseBBoxJson(cJSON* bbox, BoundingBox& out) {
    if (bbox == nullptr || !cJSON_IsObject(bbox)) return false;
    cJSON* bx = cJSON_GetObjectItem(bbox, "x");
    cJSON* by = cJSON_GetObjectItem(bbox, "y");
    cJSON* bw = cJSON_GetObjectItem(bbox, "width");
    cJSON* bh = cJSON_GetObjectItem(bbox, "height");
    if (bx && by && bw && bh) {
        out = BoundingBox(bx->valueint, by->valueint, bw->valueint, bh->valueint);
        return out.width > 0 && out.height > 0;
    }
    cJSON* x1 = cJSON_GetObjectItem(bbox, "x1");
    cJSON* y1 = cJSON_GetObjectItem(bbox, "y1");
    cJSON* x2 = cJSON_GetObjectItem(bbox, "x2");
    cJSON* y2 = cJSON_GetObjectItem(bbox, "y2");
    if (x1 && y1 && x2 && y2) {
        out = BoundingBox(x1->valueint, y1->valueint,
                          x2->valueint - x1->valueint,
                          y2->valueint - y1->valueint);
        return out.width > 0 && out.height > 0;
    }
    return false;
}

bool OnlineDetector::ParseResponse(const std::string& json_str, DetectionResult& result) {
    cJSON* json = cJSON_Parse(json_str.c_str());
    if (json == nullptr) {
        ESP_LOGE(TAG, "Failed to parse JSON response");
        return false;
    }

    // ---------------- 新协议：统一响应 {frame_id, task, result, performance} ----------------
    cJSON* result_obj = cJSON_GetObjectItem(json, "result");
    if (result_obj != nullptr && cJSON_IsObject(result_obj)) {
        cJSON* fid = cJSON_GetObjectItem(json, "frame_id");
        if (fid && cJSON_IsNumber(fid)) result.frame_id = fid->valueint;
        cJSON* tk = cJSON_GetObjectItem(json, "task");
        if (tk && cJSON_IsString(tk)) result.task = tk->valuestring;

        // performance: decode_ms / infer_ms / total_ms
        cJSON* perf = cJSON_GetObjectItem(json, "performance");
        if (perf && cJSON_IsObject(perf)) {
            cJSON* v = cJSON_GetObjectItem(perf, "decode_ms");
            if (v) result.decode_ms = (float)v->valuedouble;
            v = cJSON_GetObjectItem(perf, "infer_ms");
            if (v) result.infer_ms = (float)v->valuedouble;
            v = cJSON_GetObjectItem(perf, "total_ms");
            if (v) result.total_ms = (float)v->valuedouble;
        }

        // face: {bbox, confidence}
        cJSON* face = cJSON_GetObjectItem(result_obj, "face");
        if (face != nullptr && cJSON_IsObject(face)) {
            Detection d;
            d.class_name = "face";
            cJSON* conf = cJSON_GetObjectItem(face, "confidence");
            if (conf && cJSON_IsNumber(conf)) d.confidence = (float)conf->valuedouble;
            if (ParseBBoxJson(cJSON_GetObjectItem(face, "bbox"), d.box)) {
                result.detections.push_back(d);
            }
        }

        // emotion: {label, confidence}
        cJSON* emotion = cJSON_GetObjectItem(result_obj, "emotion");
        if (emotion != nullptr && cJSON_IsObject(emotion)) {
            cJSON* label = cJSON_GetObjectItem(emotion, "label");
            cJSON* econf = cJSON_GetObjectItem(emotion, "confidence");
            if (label && cJSON_IsString(label)) {
                result.emotion.available = true;
                result.emotion.label = label->valuestring;
                if (econf && cJSON_IsNumber(econf)) result.emotion.confidence = (float)econf->valuedouble;
            }
        }

        // posture: {state, reason, persons, analyzed_person}
        cJSON* state = cJSON_GetObjectItem(result_obj, "state");
        if (state != nullptr && cJSON_IsString(state)) {
            result.posture.available = true;
            result.posture.state = state->valuestring;
            cJSON* reason = cJSON_GetObjectItem(result_obj, "reason");
            if (reason && cJSON_IsString(reason)) result.posture.reason = reason->valuestring;
        }

        // heart_rate: {bpm, confidence, fs, frames_used} 或 {error}
        cJSON* bpm = cJSON_GetObjectItem(result_obj, "bpm");
        if (bpm != nullptr && cJSON_IsNumber(bpm)) {
            result.heart_rate.available = true;
            result.heart_rate.bpm = (float)bpm->valuedouble;
            cJSON* hconf = cJSON_GetObjectItem(result_obj, "confidence");
            if (hconf) result.heart_rate.confidence = (float)hconf->valuedouble;
            cJSON* fs = cJSON_GetObjectItem(result_obj, "fs");
            if (fs) result.heart_rate.fs = (float)fs->valuedouble;
            cJSON* fu = cJSON_GetObjectItem(result_obj, "frames_used");
            if (fu) result.heart_rate.frames_used = fu->valueint;
        } else {
            cJSON* herr = cJSON_GetObjectItem(result_obj, "error");
            if (herr && cJSON_IsString(herr)) result.heart_rate.error_message = herr->valuestring;
        }

        cJSON_Delete(json);
        return true;
    }

    // ---------------- 旧协议：{detections: [...]} 兼容 ----------------
    cJSON* detections = cJSON_GetObjectItem(json, "detections");
    if (detections == nullptr || !cJSON_IsArray(detections)) {
        cJSON_Delete(json);
        return true;
    }

    cJSON* item = nullptr;
    cJSON_ArrayForEach(item, detections) {
        Detection d;

        // class / class_name
        cJSON* cls = cJSON_GetObjectItem(item, "class");
        if (cls == nullptr) cls = cJSON_GetObjectItem(item, "class_name");
        if (cls && cJSON_IsString(cls)) {
            d.class_name = cls->valuestring;
        }

        // class_id
        cJSON* cid = cJSON_GetObjectItem(item, "class_id");
        if (cid && cJSON_IsNumber(cid)) {
            d.class_id = cid->valueint;
        }

        // confidence / score
        cJSON* conf = cJSON_GetObjectItem(item, "confidence");
        if (conf == nullptr) conf = cJSON_GetObjectItem(item, "score");
        if (conf && cJSON_IsNumber(conf)) {
            d.confidence = (float)conf->valuedouble;
        }

        // bbox
        if (ParseBBoxJson(cJSON_GetObjectItem(item, "bbox"), d.box)) {
            if (d.class_name.empty()) {
                d.class_name = "face";
            }
            result.detections.push_back(d);
        }
    }

    cJSON_Delete(json);
    return true;
}

void OnlineDetector::SetEndpoint(const std::string& url) {
    endpoint_url_ = url;
    Settings settings("vision", true);
    settings.SetString("online_url", url);
}

void OnlineDetector::SetTimeoutSec(int sec) {
    if (sec > 0) timeout_sec_ = sec;
    Settings settings("vision", true);
    settings.SetInt("online_timeout", timeout_sec_);
}

void OnlineDetector::SetJpegQuality(int quality) {
    if (quality >= 10 && quality <= 100) jpeg_quality_ = quality;
    Settings settings("vision", true);
    settings.SetInt("online_quality", jpeg_quality_);
}

void OnlineDetector::SetTask(const std::string& task) {
    if (task != kTaskFaceEmotion && task != kTaskPosture && task != kTaskHeartRate) {
        ESP_LOGW(TAG, "Unknown task '%s', ignoring", task.c_str());
        return;
    }
    task_ = task;
    Settings settings("vision", true);
    settings.SetString("online_task", task);
    ESP_LOGI(TAG, "Online detector task set to: %s", task.c_str());
}

void OnlineDetector::SetAutoFrameId(bool enabled) {
    auto_frame_id_ = enabled;
}

const std::string& OnlineDetector::GetEndpoint() const { return endpoint_url_; }
int OnlineDetector::GetTimeoutSec() const { return timeout_sec_; }
int OnlineDetector::GetJpegQuality() const { return jpeg_quality_; }
const std::string& OnlineDetector::GetTask() const { return task_; }
bool OnlineDetector::GetAutoFrameId() const { return auto_frame_id_; }
