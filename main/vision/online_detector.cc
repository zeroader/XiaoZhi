#include "online_detector.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cstring>
#include <cJSON.h>

#include "board.h"
#include "system_info.h"
#include "settings.h"
#include "image_to_jpeg.h"

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#define TAG "OnlineDetector"

// 发送前降采样宽度上限（原始帧先缩小再编码上传，大幅降低带宽占用）
// 检测在服务器端进行，小图足够：320x240 约 15~30KB，640x480 约 60~120KB
#define SEND_MAX_WIDTH 320

// JpegChunk for streaming JPEG encode -> PSRAM buffer
struct JpegChunk {
    uint8_t* data;
    size_t len;
};

// 最近邻降采样 RGB565（发送前缩小，减少上传体积；内部分配 PSRAM，需调用方释放）
// 返回新 buffer（heap_caps_malloc MALLOC_CAP_SPIRAM），失败返回 nullptr
static uint8_t* DownscaleRgb565(const uint8_t* src, int src_w, int src_h,
                                int dst_w, int dst_h) {
    if (src == nullptr || dst_w <= 0 || dst_h <= 0) return nullptr;
    size_t dst_bytes = (size_t)dst_w * dst_h * 2;
    uint8_t* dst = (uint8_t*)heap_caps_malloc(dst_bytes, MALLOC_CAP_SPIRAM);
    if (dst == nullptr) {
        ESP_LOGE(TAG, "PSRAM alloc failed for downscale (%ux%u)", dst_w, dst_h);
        return nullptr;
    }
    const uint16_t* src16 = (const uint16_t*)src;
    uint16_t* dst16 = (uint16_t*)dst;
    for (int y = 0; y < dst_h; y++) {
        int sy = std::min(src_h - 1, (y * src_h) / dst_h);
        const uint16_t* src_row = src16 + (size_t)sy * src_w;
        uint16_t* dst_row = dst16 + (size_t)y * dst_w;
        for (int x = 0; x < dst_w; x++) {
            int sx = std::min(src_w - 1, (x * src_w) / dst_w);
            dst_row[x] = src_row[sx];
        }
    }
    return dst;
}

OnlineDetector::OnlineDetector()
    : initialized_(false)
    , timeout_sec_(10)
    , jpeg_quality_(75)
    , task_(kTaskFaceEmotion)
    , frame_id_(0)
    , auto_frame_id_(true)
    , calibrate_once_(false) {
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

    // Step 1+2: 生成上传用 JPEG 字节
    // - JPEG 帧（OV2640 硬件直出）：零编码，直接引用帧缓冲
    // - RGB565 帧（旧配置）：降采样 + 软编码
    uint8_t* scaled_buf = nullptr;
    const uint8_t* enc_data = (const uint8_t*)frame.data;
    int enc_w = frame.width;
    int enc_h = frame.height;
    uint8_t* jpeg_buffer = nullptr;
    size_t jpeg_size = 0;
    bool own_jpeg_buffer = true;  // JPEG 直出时指向帧缓冲（共享帧/摄像头 fb），不释放

    if (frame.pixel_format == PIXFORMAT_JPEG) {
        // OV2640 硬件 JPEG 直出：跳过降采样与软编码，直接上传
        enc_data = (const uint8_t*)frame.data;
        enc_w = frame.width;
        enc_h = frame.height;
        jpeg_buffer = (uint8_t*)frame.data;
        jpeg_size = frame.len;
        own_jpeg_buffer = false;
    } else {
        // Step 1: 发送前降采样（仅 RGB565 且超宽时执行），减少上传体积
        if (frame.width > SEND_MAX_WIDTH) {
            enc_h = (frame.height * SEND_MAX_WIDTH) / frame.width;
            scaled_buf = DownscaleRgb565((const uint8_t*)frame.data, frame.width, frame.height,
                                         SEND_MAX_WIDTH, enc_h);
            if (scaled_buf != nullptr) {
                enc_data = scaled_buf;
                enc_w = SEND_MAX_WIDTH;
            } else {
                // 降采样失败则退回原始帧
                enc_w = frame.width;
                enc_h = frame.height;
            }
        }

        // Step 2: JPEG encode the (possibly downscaled) RGB565 frame
        QueueHandle_t jpeg_queue = xQueueCreate(40, sizeof(JpegChunk));
        if (jpeg_queue == nullptr) {
            if (scaled_buf) heap_caps_free(scaled_buf);
            ESP_LOGE(TAG, "Failed to create JPEG queue");
            return result;
        }

        std::thread encoder_thread([enc_data, frame, enc_w, enc_h, jpeg_queue, this]() {
            image_to_jpeg_cb((uint8_t*)enc_data, (size_t)enc_w * enc_h * 2, enc_w, enc_h,
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
        if (scaled_buf) heap_caps_free(scaled_buf);   // 释放降采样临时缓冲
        jpeg_size = jpeg_total;
    }

    if (jpeg_buffer == nullptr || jpeg_size == 0) {
        ESP_LOGE(TAG, "JPEG encode failed");
        if (own_jpeg_buffer && jpeg_buffer) heap_caps_free(jpeg_buffer);
        return result;
    }

    auto t_encode = esp_timer_get_time();
    ESP_LOGI(TAG, "Encoded: %dx%d -> %dx%d jpeg=%d B (encode=%dms)",
             frame.width, frame.height, enc_w, enc_h,
             (int)jpeg_size, (int)((t_encode - total_start) / 1000LL));

    // Step 3: multipart/form-data 二进制上传（免 base64）
    std::string boundary = "----VISION_ONLINE_DETECT_BOUNDARY";

    auto network = Board::GetInstance().GetNetwork();
    auto http = network->CreateHttp();
    // CreateHttp's argument is a connection ID, not a timeout.  Without this
    // explicit timeout the client waits for its 30-second default, making a
    // short LAN outage look like a permanently stalled detector.
    http->SetTimeout(timeout_sec_ * 1000);

    http->SetHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    http->SetHeader("Transfer-Encoding", "chunked");
    http->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());

    if (!http->Open("POST", endpoint_url_)) {
        ESP_LOGE(TAG, "Failed to open HTTP connection to %s", endpoint_url_.c_str());
        result.connection_ok = false;
        result.error_message = "cannot connect to detection server";
        http->Close();
        if (own_jpeg_buffer) heap_caps_free(jpeg_buffer);
        return result;
    }

    // 表单字段：frame_id / task / capture_timestamp_ms / calibrate / width / height
    std::string meta_header;
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"frame_id\"\r\n\r\n";
    meta_header += std::to_string(frame_id_) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"task\"\r\n\r\n";
    meta_header += task_ + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"capture_timestamp_ms\"\r\n\r\n";
    meta_header += std::to_string((uint64_t)(total_start / 1000LL)) + "\r\n";
    if (calibrate_once_ && task_ == kTaskPosture) {
        meta_header += "--" + boundary + "\r\n";
        meta_header += "Content-Disposition: form-data; name=\"calibrate\"\r\n\r\n";
        meta_header += "true\r\n";
        calibrate_once_ = false;  // 一次性标定请求
    }
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"width\"\r\n\r\n";
    meta_header += std::to_string(enc_w) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"height\"\r\n\r\n";
    meta_header += std::to_string(enc_h) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n";
    meta_header += "Content-Type: image/jpeg\r\n\r\n";
    http->Write(meta_header.c_str(), meta_header.size());

    // JPEG 二进制
    http->Write((const char*)jpeg_buffer, jpeg_size);
    if (own_jpeg_buffer) heap_caps_free(jpeg_buffer);

    std::string footer = "\r\n--" + boundary + "--\r\n";
    http->Write(footer.c_str(), footer.size());
    http->Write("", 0);

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

    // Step 4: Parse response
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
        if (result.task == kTaskHeartRate && bpm != nullptr && cJSON_IsNumber(bpm)) {
            result.heart_rate.available = true;
            result.heart_rate.bpm = (float)bpm->valuedouble;
            cJSON* hconf = cJSON_GetObjectItem(result_obj, "confidence");
            if (hconf) result.heart_rate.confidence = (float)hconf->valuedouble;
            cJSON* fs = cJSON_GetObjectItem(result_obj, "fs");
            if (fs) result.heart_rate.fs = (float)fs->valuedouble;
            cJSON* fu = cJSON_GetObjectItem(result_obj, "frames_used");
            if (fu) result.heart_rate.frames_used = fu->valueint;
        } else if (result.task == kTaskHeartRate) {
            cJSON* herr = cJSON_GetObjectItem(result_obj, "error");
            if (herr && cJSON_IsString(herr)) result.heart_rate.error_message = herr->valuestring;
        }

        if (result.task == kTaskBloodPressure) {
            result.blood_pressure.available = true;
            cJSON* status = cJSON_GetObjectItem(result_obj, "status");
            if (status && cJSON_IsString(status)) result.blood_pressure.status = status->valuestring;
            cJSON* quality = cJSON_GetObjectItem(result_obj, "quality");
            if (quality && cJSON_IsObject(quality)) {
                cJSON* value = cJSON_GetObjectItem(quality, "duration_s");
                if (value && cJSON_IsNumber(value)) result.blood_pressure.duration_s = (float)value->valuedouble;
                value = cJSON_GetObjectItem(quality, "required_window_s");
                if (value && cJSON_IsNumber(value)) result.blood_pressure.required_window_s = (float)value->valuedouble;
                value = cJSON_GetObjectItem(quality, "reason");
                if (value && cJSON_IsString(value)) result.blood_pressure.reason = value->valuestring;
            }
            cJSON* sbp = cJSON_GetObjectItem(result_obj, "sbp_mmHg");
            cJSON* dbp = cJSON_GetObjectItem(result_obj, "dbp_mmHg");
            if (sbp && dbp && cJSON_IsNumber(sbp) && cJSON_IsNumber(dbp)) {
                result.blood_pressure.sbp_mmHg = (float)sbp->valuedouble;
                result.blood_pressure.dbp_mmHg = (float)dbp->valuedouble;
                result.blood_pressure.ready = true;
            }
            if (result.blood_pressure.reason.empty()) {
                cJSON* error = cJSON_GetObjectItem(result_obj, "error");
                if (error && cJSON_IsString(error)) result.blood_pressure.reason = error->valuestring;
            }
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
    if (task != kTaskFaceEmotion && task != kTaskPosture && task != kTaskHeartRate &&
        task != kTaskBloodPressure) {
        ESP_LOGW(TAG, "Unknown task '%s', ignoring", task.c_str());
        return;
    }
    if (task_ == task) {
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

void OnlineDetector::SetCalibrateOnce(bool enabled) {
    calibrate_once_ = enabled;
    ESP_LOGI(TAG, "Online detector posture calibration %s",
             enabled ? "armed (next posture request)" : "cancelled");
}

const std::string& OnlineDetector::GetEndpoint() const { return endpoint_url_; }
int OnlineDetector::GetTimeoutSec() const { return timeout_sec_; }
int OnlineDetector::GetJpegQuality() const { return jpeg_quality_; }
const std::string& OnlineDetector::GetTask() const { return task_; }
bool OnlineDetector::GetAutoFrameId() const { return auto_frame_id_; }
