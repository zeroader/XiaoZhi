#include "remote_detector.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cJSON.h>

#include "board.h"
#include "system_info.h"
#include "settings.h"
#include "jpg/image_to_jpeg.h"

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#define TAG "RemoteDetector"

struct JpegChunk {
    uint8_t* data;
    size_t len;
};

RemoteDetector::RemoteDetector()
    : initialized_(false)
    , timeout_sec_(10) {
}

RemoteDetector::~RemoteDetector() {
    Deinitialize();
}

const char* RemoteDetector::GetName() const {
    return "remote";
}

bool RemoteDetector::Initialize() {
    if (initialized_) {
        return true;
    }

    // Load persisted configuration from NVS (saved by SetEndpoint/SetAuthToken/SetRequestTimeoutSec)
    {
        Settings settings("vision");
        std::string url = settings.GetString("remote_url");
        if (!url.empty()) {
            endpoint_url_ = url;
        }
        std::string token = settings.GetString("remote_token");
        if (!token.empty()) {
            auth_token_ = token;
        }
        int32_t timeout = settings.GetInt("remote_timeout", 0);
        if (timeout > 0) {
            timeout_sec_ = timeout;
        }
    }

    if (endpoint_url_.empty()) {
        ESP_LOGW(TAG, "Remote detector endpoint URL not set - Initialize() will succeed but Detect() will fail");
    }
    initialized_ = true;
    ESP_LOGI(TAG, "Remote detector initialized, endpoint=%s, timeout=%ds",
             endpoint_url_.empty() ? "(not set)" : endpoint_url_.c_str(),
             timeout_sec_);
    return true;
}

void RemoteDetector::Deinitialize() {
    if (!initialized_) return;
    initialized_ = false;
    ESP_LOGI(TAG, "Remote detector deinitialized");
}

DetectionResult RemoteDetector::Detect(const ImageFrame& frame) {
    DetectionResult result;
    result.source_width = frame.width;
    result.source_height = frame.height;
    result.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000LL);

    if (!initialized_) {
        ESP_LOGE(TAG, "Remote detector not initialized");
        return result;
    }
    if (endpoint_url_.empty()) {
        ESP_LOGE(TAG, "Remote detector endpoint URL is not set");
        return result;
    }
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) {
        ESP_LOGE(TAG, "Invalid frame for remote detection");
        return result;
    }

    auto total_start = esp_timer_get_time();

    QueueHandle_t jpeg_queue = xQueueCreate(40, sizeof(JpegChunk));
    if (jpeg_queue == nullptr) {
        ESP_LOGE(TAG, "Failed to create JPEG queue");
        return result;
    }

    std::thread encoder_thread([&frame, jpeg_queue]() {
        image_to_jpeg_cb(const_cast<uint8_t*>(frame.data), frame.len, frame.width, frame.height,
            static_cast<pixformat_t>(frame.pixel_format), 80,
            [](void* arg, size_t index, const void* data, size_t len) -> size_t {
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

    auto network = Board::GetInstance().GetNetwork();
    auto http = network->CreateHttp(timeout_sec_);

    std::string boundary = "----VISION_REMOTE_DETECT_BOUNDARY";

    http->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());
    http->SetHeader("Client-Id", Board::GetInstance().GetUuid().c_str());
    if (!auth_token_.empty()) {
        http->SetHeader("Authorization", "Bearer " + auth_token_);
    }
    http->SetHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    http->SetHeader("Transfer-Encoding", "chunked");

    if (!http->Open("POST", endpoint_url_)) {
        ESP_LOGE(TAG, "Failed to open HTTP connection to %s", endpoint_url_.c_str());
        encoder_thread.join();
        JpegChunk chunk;
        while (xQueueReceive(jpeg_queue, &chunk, 0) == pdPASS) {
            if (chunk.data != nullptr) heap_caps_free(chunk.data);
        }
        vQueueDelete(jpeg_queue);
        return result;
    }

    std::string meta_header;
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"width\"\r\n\r\n";
    meta_header += std::to_string(frame.width) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"height\"\r\n\r\n";
    meta_header += std::to_string(frame.height) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n";
    meta_header += "Content-Type: image/jpeg\r\n\r\n";
    http->Write(meta_header.c_str(), meta_header.size());

    size_t total_jpeg = 0;
    while (true) {
        JpegChunk chunk;
        if (xQueueReceive(jpeg_queue, &chunk, pdMS_TO_TICKS(5000)) != pdPASS) {
            ESP_LOGE(TAG, "Timeout receiving JPEG chunk from encoder");
            break;
        }
        if (chunk.data == nullptr) {
            break;
        }
        http->Write((const char*)chunk.data, chunk.len);
        total_jpeg += chunk.len;
        heap_caps_free(chunk.data);
    }
    encoder_thread.join();
    vQueueDelete(jpeg_queue);

    std::string footer = "\r\n--" + boundary + "--\r\n";
    http->Write(footer.c_str(), footer.size());
    http->Write("", 0);

    if (http->GetStatusCode() != 200) {
        ESP_LOGE(TAG, "Remote detect HTTP status=%d", http->GetStatusCode());
        http->Close();
        return result;
    }

    std::string response = http->ReadAll();
    http->Close();

    auto parse_start = esp_timer_get_time();
    ParseResponseJson(response, result);
    auto parse_end = esp_timer_get_time();

    auto total_end = esp_timer_get_time();
    ESP_LOGI(TAG, "Remote detect: jpeg=%uB, detections=%d, total=%dms (http+enc=%dms, parse=%dms)",
             (unsigned)total_jpeg,
             (int)result.detections.size(),
             (int)((total_end - total_start) / 1000LL),
             (int)((parse_start - total_start) / 1000LL),
             (int)((parse_end - parse_start) / 1000LL));

    return result;
}

bool RemoteDetector::ParseResponseJson(const std::string& json_str, DetectionResult& result) {
    cJSON* json = cJSON_Parse(json_str.c_str());
    if (json == nullptr) {
        ESP_LOGE(TAG, "Failed to parse remote detect response JSON");
        return false;
    }

    cJSON* detections = cJSON_GetObjectItem(json, "detections");
    if (detections == nullptr || !cJSON_IsArray(detections)) {
        cJSON_Delete(json);
        return true;
    }

    cJSON* item = nullptr;
    cJSON_ArrayForEach(item, detections) {
        Detection d;

        cJSON* class_name = cJSON_GetObjectItem(item, "class_name");
        if (class_name && cJSON_IsString(class_name)) {
            d.class_name = class_name->valuestring;
        }

        cJSON* class_id = cJSON_GetObjectItem(item, "class_id");
        if (class_id && cJSON_IsNumber(class_id)) {
            d.class_id = class_id->valueint;
        } else {
            d.class_id = -1;
        }

        cJSON* conf = cJSON_GetObjectItem(item, "confidence");
        if (conf && cJSON_IsNumber(conf)) {
            d.confidence = (float)conf->valuedouble;
        }

        cJSON* bbox = cJSON_GetObjectItem(item, "bbox");
        if (bbox) {
            cJSON* bx = cJSON_GetObjectItem(bbox, "x");
            cJSON* by = cJSON_GetObjectItem(bbox, "y");
            cJSON* bw = cJSON_GetObjectItem(bbox, "width");
            cJSON* bh = cJSON_GetObjectItem(bbox, "height");
            if (bx && by && bw && bh) {
                d.box = BoundingBox(bx->valueint, by->valueint,
                                    bw->valueint, bh->valueint);
            } else {
                cJSON* x1 = cJSON_GetObjectItem(bbox, "x1");
                cJSON* y1 = cJSON_GetObjectItem(bbox, "y1");
                cJSON* x2 = cJSON_GetObjectItem(bbox, "x2");
                cJSON* y2 = cJSON_GetObjectItem(bbox, "y2");
                if (x1 && y1 && x2 && y2) {
                    d.box = BoundingBox(x1->valueint, y1->valueint,
                                        x2->valueint - x1->valueint,
                                        y2->valueint - y1->valueint);
                }
            }
        }

        if (d.box.width > 0 && d.box.height > 0) {
            if (d.class_name.empty()) {
                d.class_name = "obj_" + std::to_string(d.class_id);
            }
            result.detections.push_back(d);
        }
    }

    cJSON_Delete(json);
    return true;
}

void RemoteDetector::SetEndpoint(const std::string& url) {
    endpoint_url_ = url;
    Settings settings("vision", true);
    settings.SetString("remote_url", url);
}

void RemoteDetector::SetAuthToken(const std::string& token) {
    auth_token_ = token;
    Settings settings("vision", true);
    settings.SetString("remote_token", token);
}

void RemoteDetector::SetRequestTimeoutSec(int sec) {
    if (sec > 0) timeout_sec_ = sec;
    Settings settings("vision", true);
    settings.SetInt("remote_timeout", timeout_sec_);
}

void RemoteDetector::SetExtraHeader(const std::string& key, const std::string& value) {
    extra_headers_ += key + ": " + value + "\r\n";
}

const std::string& RemoteDetector::GetEndpoint() const {
    return endpoint_url_;
}

int RemoteDetector::GetRequestTimeoutSec() const {
    return timeout_sec_;
}
