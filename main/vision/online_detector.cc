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

#define TAG "OnlineDetector"

// 发送前降采样宽度上限（原始帧先缩小再编码上传，大幅降低带宽占用）
// 检测在服务器端进行，小图足够：320x240 约 15~30KB，640x480 约 60~120KB
#define SEND_MAX_WIDTH 320

namespace {

struct JpegOutputContext {
    OnlineDetector* detector;
    size_t size;
    bool failed;
};

constexpr size_t kInitialJpegBufferCapacity = 32 * 1024;
constexpr size_t kMaxJpegBufferCapacity = 256 * 1024;

}  // namespace

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

size_t OnlineDetector::JpegOutputCallback(void* arg, size_t index,
                                           const void* data, size_t len) {
    auto* context = static_cast<JpegOutputContext*>(arg);
    if (context == nullptr) return 0;
    if (context->failed) return 0;
    if (context->detector == nullptr || data == nullptr) {
        context->failed = true;
        return 0;
    }

    const size_t written = context->detector->AppendJpegData(index, data, len);
    if (written != len) {
        context->failed = true;
        return 0;
    }
    context->size = index + written;
    return written;
}

bool OnlineDetector::EnsureJpegBuffer(size_t required) {
    if (required <= jpeg_buffer_cap_) return true;
    if (required > kMaxJpegBufferCapacity) {
        ESP_LOGE(TAG, "JPEG output exceeds reusable buffer limit: %u B", (unsigned)required);
        return false;
    }

    size_t new_capacity = jpeg_buffer_cap_ == 0
        ? kInitialJpegBufferCapacity : jpeg_buffer_cap_;
    while (new_capacity < required) {
        if (new_capacity > kMaxJpegBufferCapacity / 2) {
            new_capacity = kMaxJpegBufferCapacity;
            break;
        }
        new_capacity *= 2;
    }

    uint8_t* new_buffer = (uint8_t*)heap_caps_aligned_alloc(
        16, new_capacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (new_buffer == nullptr) {
        ESP_LOGE(TAG, "PSRAM alloc failed for reusable JPEG buffer (%u B)",
                 (unsigned)new_capacity);
        return false;
    }
    if (jpeg_buffer_ != nullptr && jpeg_buffer_cap_ != 0) {
        memcpy(new_buffer, jpeg_buffer_, jpeg_buffer_cap_);
        heap_caps_free(jpeg_buffer_);
    }
    jpeg_buffer_ = new_buffer;
    jpeg_buffer_cap_ = new_capacity;
    return true;
}

size_t OnlineDetector::AppendJpegData(size_t index, const void* data, size_t len) {
    if (data == nullptr || len == 0) return len;
    if (index > SIZE_MAX - len || !EnsureJpegBuffer(index + len)) return 0;
    memcpy(jpeg_buffer_ + index, data, len);
    return len;
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
    if (jpeg_buffer_ != nullptr) {
        heap_caps_free(jpeg_buffer_);
        jpeg_buffer_ = nullptr;
        jpeg_buffer_cap_ = 0;
    }
    ESP_LOGI(TAG, "Online detector deinitialized");
}

DetectionResult OnlineDetector::Detect(const ImageFrame& frame) {
    return Detect(frame, std::vector<std::string>{task_});
}

DetectionResult OnlineDetector::Detect(const ImageFrame& frame,
                                       const std::vector<std::string>& tasks) {
    DetectionResult result;
    result.source_width = frame.width;
    result.source_height = frame.height;
    result.timestamp_ms = (uint64_t)(esp_timer_get_time() / 1000LL);
    const std::vector<std::string> selected_tasks = tasks.empty()
        ? std::vector<std::string>{task_} : tasks;
    std::string task_list;
    for (size_t i = 0; i < selected_tasks.size(); ++i) {
        if (i != 0) task_list += ',';
        task_list += selected_tasks[i];
    }
    result.task = task_list;
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

    if (frame.pixel_format == PIXFORMAT_JPEG) {
        // OV2640 硬件 JPEG 直出：跳过降采样与软编码，直接上传
        enc_data = (const uint8_t*)frame.data;
        enc_w = frame.width;
        enc_h = frame.height;
        jpeg_buffer = (uint8_t*)frame.data;
        jpeg_size = frame.len;
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

        // Step 2: 在当前检测线程内同步编码。编码器本身只需短暂工作空间，
        // 输出直接写入可复用的 PSRAM 缓冲，避免每帧创建线程、队列和 chunk。
        JpegOutputContext output = { .detector = this, .size = 0, .failed = false };
        const bool encoded = image_to_jpeg_cb(
            (uint8_t*)enc_data, (size_t)enc_w * enc_h * 2, enc_w, enc_h,
            (pixformat_t)frame.pixel_format, jpeg_quality_,
            &OnlineDetector::JpegOutputCallback, &output);
        if (scaled_buf) heap_caps_free(scaled_buf);
        if (!encoded || output.failed || output.size == 0) {
            ESP_LOGE(TAG, "JPEG encode failed (callback=%s, size=%u)",
                     encoded ? "ok" : "error", (unsigned)output.size);
            return result;
        }
        jpeg_buffer = jpeg_buffer_;
        jpeg_size = output.size;
    }

    if (jpeg_buffer == nullptr || jpeg_size == 0) {
        ESP_LOGE(TAG, "JPEG encode failed");
        return result;
    }

    auto t_encode = esp_timer_get_time();
    ESP_LOGI(TAG, "Encoded: %dx%d -> %dx%d jpeg=%d B (encode=%dms)",
             frame.width, frame.height, enc_w, enc_h,
             (int)jpeg_size, (int)((t_encode - total_start) / 1000LL));

    // Step 3: multipart/form-data 二进制上传（免 base64）
    std::string boundary = "----VISION_ONLINE_DETECT_BOUNDARY";

    // 表单字段：frame_id / task(s) / capture_timestamp_ms / calibrate / width / height
    std::string meta_header;
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"frame_id\"\r\n\r\n";
    meta_header += std::to_string(frame_id_) + "\r\n";
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"task\"\r\n\r\n";
    meta_header += selected_tasks.front() + "\r\n";
    if (selected_tasks.size() > 1) {
        meta_header += "--" + boundary + "\r\n";
        meta_header += "Content-Disposition: form-data; name=\"tasks\"\r\n\r\n";
        meta_header += task_list + "\r\n";
    }
    meta_header += "--" + boundary + "\r\n";
    meta_header += "Content-Disposition: form-data; name=\"capture_timestamp_ms\"\r\n\r\n";
    meta_header += std::to_string((uint64_t)(total_start / 1000LL)) + "\r\n";
    if (calibrate_once_ && std::find(selected_tasks.begin(), selected_tasks.end(),
                                     kTaskPosture) != selected_tasks.end()) {
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
    std::string footer = "\r\n--" + boundary + "--\r\n";

    auto network = Board::GetInstance().GetNetwork();
    auto http = network->CreateHttp(0);
    if (http == nullptr) {
        ESP_LOGE(TAG, "Failed to create HTTP client");
        result.connection_ok = false;
        result.error_message = "cannot create HTTP client";
        return result;
    }
    http->SetTimeout(timeout_sec_ * 1000);
    http->SetHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    http->SetHeader("Content-Length", std::to_string(meta_header.size() + jpeg_size + footer.size()));
    http->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());

    if (!http->Open("POST", endpoint_url_)) {
        ESP_LOGE(TAG, "Failed to open HTTP connection to %s", endpoint_url_.c_str());
        result.connection_ok = false;
        result.error_message = "cannot connect to detection server";
        http->Close();
        return result;
    }

    auto write_part = [&http](const char* data, size_t size) {
        return http->Write(data, size) == static_cast<int>(size);
    };
    if (!write_part(meta_header.c_str(), meta_header.size()) ||
        !write_part((const char*)jpeg_buffer, jpeg_size) ||
        !write_part(footer.c_str(), footer.size())) {
        ESP_LOGE(TAG, "Failed to send detection request body");
        result.connection_ok = false;
        result.error_message = "failed to send detection request";
        http->Close();
        return result;
    }

    const int status_code = http->GetStatusCode();
    if (status_code != 200) {
        const std::string error_response = status_code > 0 ? http->ReadAll() : std::string();
        ESP_LOGE(TAG, "HTTP status=%d, response=%s", status_code,
                 error_response.empty() ? "(empty)" : error_response.c_str());
        result.connection_ok = false;
        result.error_message = "detection server returned error " + std::to_string(status_code);
        http->Close();
        return result;
    }

    std::string response = http->ReadAll();
    http->Close();

    auto t_http = esp_timer_get_time();

    // Step 4: Parse response
    ParseResponse(response, result);

    auto total_end = esp_timer_get_time();
    ESP_LOGI(TAG, "Detect: tasks=%s frame=%d jpeg=%d B, detections=%d, total=%dms (encode=%dms, http=%dms) server(dec=%.1fms,inf=%.1fms,total=%.1fms)",
             task_list.c_str(),
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

        const bool includes_heart_rate = result.task.find(kTaskHeartRate) != std::string::npos;
        const bool includes_blood_pressure =
            result.task.find(kTaskBloodPressure) != std::string::npos;

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
        } else if (includes_heart_rate) {
            cJSON* herr = cJSON_GetObjectItem(result_obj, "error");
            if (herr && cJSON_IsString(herr)) result.heart_rate.error_message = herr->valuestring;
        }

        if (includes_blood_pressure) {
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
