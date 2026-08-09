#include "vision_pipeline.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cJSON.h>

#include "board.h"
#include "esp32_camera.h"
#include "lvgl_display.h"
#include "mcp_server.h"
#include "system_info.h"
#include "application.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define TAG "VisionPipeline"

PipelineStats::PipelineStats() { Reset(); }

void PipelineStats::Reset() {
    capture_count = 0;
    detect_count = 0;
    display_count = 0;
    total_capture_ms = 0;
    total_detect_ms = 0;
    total_compose_ms = 0;
    total_display_ms = 0;
    last_detection_count = 0;
    loop_period_ms = 0;
}

VisionPipeline::VisionPipeline()
    : initialized_(false)
    , has_camera_(false)
    , active_detector_type_(DetectorType::kNone)
    , active_detector_(nullptr)
    , continuous_running_(false)
    , continuous_period_ms_(500)
    , continuous_task_(kTaskFaceEmotion) {
}

VisionPipeline::~VisionPipeline() {
    Deinitialize();
}

VisionPipeline& VisionPipeline::GetInstance() {
    static VisionPipeline instance;
    return instance;
}

Esp32Camera* VisionPipeline::ResolveEsp32Camera() {
    auto* board_camera = Board::GetInstance().GetCamera();
    if (board_camera == nullptr) return nullptr;
    return dynamic_cast<Esp32Camera*>(board_camera);
}

LvglDisplay* VisionPipeline::ResolveLvglDisplay() {
    auto* display = Board::GetInstance().GetDisplay();
    if (display == nullptr) return nullptr;
    return dynamic_cast<LvglDisplay*>(display);
}

bool VisionPipeline::Initialize() {
    if (initialized_) return true;

    ESP_LOGI(TAG, "Initializing vision pipeline");

    auto* esp32_cam = ResolveEsp32Camera();
    has_camera_ = (esp32_cam != nullptr);
    if (!has_camera_) {
        ESP_LOGW(TAG, "Board has no Esp32Camera - pipeline will be limited");
    } else {
        ESP_LOGI(TAG, "Found Esp32Camera on board");
    }

#ifndef VISION_DISABLE_LOCAL_FACE
    face_detector_ = std::make_unique<FaceDetector>();
    if (!face_detector_->Initialize()) {
        ESP_LOGW(TAG, "Face detector initialize returned false");
    }
#endif

    remote_detector_ = std::make_unique<RemoteDetector>();
    if (!remote_detector_->Initialize()) {
        ESP_LOGW(TAG, "Remote detector initialize returned false");
    }

    online_detector_ = std::make_unique<OnlineDetector>();
    if (!online_detector_->Initialize()) {
        ESP_LOGW(TAG, "Online detector initialize returned false");
    }

    display_composer_ = std::make_unique<VisionDisplay>();

    active_detector_type_ = DetectorType::kOnline;
    active_detector_ = online_detector_.get();

    initialized_ = true;
    ESP_LOGI(TAG, "Vision pipeline initialized (active=%s, camera=%s)",
             active_detector_->GetName(),
             has_camera_ ? "ok" : "none");
    return true;
}

void VisionPipeline::Deinitialize() {
    if (!initialized_) return;

    StopContinuous();

#ifndef VISION_DISABLE_LOCAL_FACE
    if (face_detector_) {
        face_detector_->Deinitialize();
    }
#endif
    if (remote_detector_) {
        remote_detector_->Deinitialize();
    }
    if (online_detector_) {
        online_detector_->Deinitialize();
    }

    active_detector_ = nullptr;
    active_detector_type_ = DetectorType::kNone;
    initialized_ = false;
    ESP_LOGI(TAG, "Vision pipeline deinitialized");
}

Detector* VisionPipeline::GetDetector(DetectorType type) {
    switch (type) {
#ifndef VISION_DISABLE_LOCAL_FACE
        case DetectorType::kFace:   return face_detector_.get();
#endif
        case DetectorType::kRemote: return remote_detector_.get();
        case DetectorType::kOnline: return online_detector_.get();
        default: return nullptr;
    }
}

DetectorType VisionPipeline::GetActiveDetectorType() const { return active_detector_type_; }

// 若活动检测器是 OnlineDetector 且 task 合法，切换其任务
static bool ApplyTaskToOnlineDetector(Detector* detector, const std::string& task) {
    auto* od = dynamic_cast<OnlineDetector*>(detector);
    if (od == nullptr || task.empty()) return false;
    if (task == kTaskAutoSchedule) return false;  // auto 仅在连续模式调度中使用
    od->SetTask(task);
    return true;
}

bool VisionPipeline::SetActiveDetector(DetectorType type) {
    Detector* d = GetDetector(type);
    if (d == nullptr) {
        ESP_LOGE(TAG, "Invalid detector type: %d", (int)type);
        return false;
    }
    active_detector_type_ = type;
    active_detector_ = d;
    ESP_LOGI(TAG, "Active detector changed to: %s", d->GetName());
    return true;
}

#ifndef VISION_DISABLE_LOCAL_FACE
FaceDetector* VisionPipeline::GetFaceDetector() { return face_detector_.get(); }
#endif
RemoteDetector* VisionPipeline::GetRemoteDetector() { return remote_detector_.get(); }
OnlineDetector* VisionPipeline::GetOnlineDetector() { return online_detector_.get(); }
VisionDisplay* VisionPipeline::GetDisplayComposer() { return display_composer_.get(); }

bool VisionPipeline::CaptureFrame(ImageFrame& out_frame) {
    auto* cam = ResolveEsp32Camera();
    if (cam == nullptr) {
        ESP_LOGE(TAG, "No Esp32Camera available for capture");
        return false;
    }
    auto t0 = esp_timer_get_time();
    if (!cam->Capture()) {
        ESP_LOGE(TAG, "Camera Capture() failed");
        return false;
    }
    auto t1 = esp_timer_get_time();
    stats_.total_capture_ms += (uint64_t)((t1 - t0) / 1000LL);
    stats_.capture_count++;

    camera_fb_t* fb = cam->GetLastCapturedFrame();
    if (fb == nullptr) {
        ESP_LOGE(TAG, "Captured frame buffer is null");
        return false;
    }
    out_frame.data = fb->buf;
    out_frame.len = fb->len;
    out_frame.width = fb->width;
    out_frame.height = fb->height;
    out_frame.pixel_format = (int)fb->format;
    return true;
}

bool VisionPipeline::ReleaseCurrentFrame() {
    auto* cam = ResolveEsp32Camera();
    if (cam == nullptr) return false;
    cam->ReleaseLastCapturedFrame();
    return true;
}

camera_fb_t* VisionPipeline::GetCameraFb() {
    auto* cam = ResolveEsp32Camera();
    return cam ? cam->GetLastCapturedFrame() : nullptr;
}

bool VisionPipeline::OneShotDetect(bool show_on_lcd, std::string* out_debug_info, const std::string& task) {
    if (!initialized_) {
        ESP_LOGE(TAG, "Pipeline not initialized");
        if (out_debug_info) *out_debug_info = "pipeline not initialized";
        return false;
    }
    if (active_detector_ == nullptr) {
        ESP_LOGE(TAG, "No active detector");
        if (out_debug_info) *out_debug_info = "no active detector";
        return false;
    }

    // 可选：切换 online 检测器任务（face_emotion / posture / heart_rate）
    ApplyTaskToOnlineDetector(active_detector_, task);

    ImageFrame frame;
    auto t_all = esp_timer_get_time();
    if (!CaptureFrame(frame)) {
        if (out_debug_info) *out_debug_info = "capture failed";
        return false;
    }

    auto t_detect0 = esp_timer_get_time();
    DetectionResult result = active_detector_->Detect(frame);
    auto t_detect1 = esp_timer_get_time();
    stats_.total_detect_ms += (uint64_t)((t_detect1 - t_detect0) / 1000LL);
    stats_.detect_count++;
    stats_.last_detection_count = (uint32_t)result.detections.size();

    last_result_ = result;
    CacheSensingResult(result);
    MaybeNotifyPosture(result);

    bool displayed = false;
    if (show_on_lcd && display_composer_) {
        auto* lcd = ResolveLvglDisplay();
        if (lcd != nullptr) {
            auto t_comp0 = esp_timer_get_time();
            auto image = display_composer_->ComposePreview(frame.data, frame.width, frame.height,
                                                          (size_t)frame.width * 2, result);
            auto t_comp1 = esp_timer_get_time();
            stats_.total_compose_ms += (uint64_t)((t_comp1 - t_comp0) / 1000LL);

            if (image) {
                auto t_disp0 = esp_timer_get_time();
                lcd->SetPreviewImage(std::move(image));
                auto t_disp1 = esp_timer_get_time();
                stats_.total_display_ms += (uint64_t)((t_disp1 - t_disp0) / 1000LL);
                stats_.display_count++;
                displayed = true;
            }
        }
    }

    ReleaseCurrentFrame();

    auto t_all1 = esp_timer_get_time();
    char buf[256];
    snprintf(buf, sizeof(buf),
             "detector=%s, detections=%d, dimensions=%dx%d, total=%dms, detect=%dms, shown=%s",
             active_detector_->GetName(),
             (int)result.detections.size(),
             frame.width, frame.height,
             (int)((t_all1 - t_all) / 1000LL),
             (int)((t_detect1 - t_detect0) / 1000LL),
             displayed ? "true" : "false");
    ESP_LOGI(TAG, "OneShotDetect: %s", buf);
    if (out_debug_info) *out_debug_info = buf;
    return true;
}

// ---------- Two-thread architecture: Preview (15 FPS) + Detect (async) ----------

void VisionPipeline::PreviewLoop() {
    ESP_LOGI(TAG, "Preview loop started (15 FPS)");

    constexpr uint32_t kPreviewPeriodMs = 67;  // ~15 FPS
    uint64_t last_iter = 0;

    while (continuous_running_.load()) {
        auto t0 = esp_timer_get_time();
        stats_.loop_period_ms = last_iter == 0 ? 0 : (uint32_t)((t0 - last_iter) / 1000LL);
        last_iter = t0;

        // Step 1: Capture frame
        ImageFrame frame;
        if (CaptureFrame(frame)) {
            // Step 2: Copy to shared buffer for detection thread
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                shared_frame_.assign(frame.data, frame.data + frame.len);
                shared_frame_w_ = frame.width;
                shared_frame_h_ = frame.height;
                shared_frame_version_++;
            }

            // Step 3: Compose preview with latest bbox (if any)
            if (display_composer_) {
                auto* lcd = ResolveLvglDisplay();
                if (lcd != nullptr) {
                    DetectionResult current_result;
                    {
                        std::lock_guard<std::mutex> lock(result_mutex_);
                        current_result = last_result_;
                    }
                    // 合并跨帧缓存的坐姿/心率，保证 LCD 上持续显示（不被每帧 face_emotion 覆盖）
                    {
                        std::lock_guard<std::mutex> lock(sensing_mutex_);
                        current_result.posture = latest_posture_;
                        current_result.heart_rate = latest_heart_rate_;
                    }
                    auto t_comp0 = esp_timer_get_time();
                    auto image = display_composer_->ComposePreview(
                        frame.data, frame.width, frame.height,
                        (size_t)frame.width * 2, current_result);
                    auto t_comp1 = esp_timer_get_time();
                    stats_.total_compose_ms += (uint64_t)((t_comp1 - t_comp0) / 1000LL);

                    if (image) {
                        auto t_disp0 = esp_timer_get_time();
                        lcd->SetPreviewImage(std::move(image));
                        auto t_disp1 = esp_timer_get_time();
                        stats_.total_display_ms += (uint64_t)((t_disp1 - t_disp0) / 1000LL);
                        stats_.display_count++;
                    }
                }
            }

            // Step 4: Release frame
            ReleaseCurrentFrame();
        }

        // Step 5: Sleep to maintain FPS
        auto t1 = esp_timer_get_time();
        int64_t elapsed = (t1 - t0) / 1000LL;
        int64_t wait = (int64_t)kPreviewPeriodMs - elapsed;
        if (wait > 0 && continuous_running_.load()) {
            vTaskDelay(pdMS_TO_TICKS((uint32_t)wait));
        }
    }
    ESP_LOGI(TAG, "Preview loop stopped");
}

void VisionPipeline::DetectionLoop() {
    ESP_LOGI(TAG, "Detection loop started, task=%s", continuous_task_.c_str());

    int consecutive_errors = 0;
    uint32_t last_version = 0;  // avoid re-processing same frame
    uint64_t last_posture_ms = 0;  // 距上次 posture 任务的毫秒数
    uint64_t last_heart_ms = 0;    // 距上次 heart_rate 任务的毫秒数

    while (continuous_running_.load()) {
        // Wait for a NEW shared frame (skip if same as last processed)
        ImageFrame detect_frame;
        uint32_t current_version;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (shared_frame_.empty()) {
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            current_version = shared_frame_version_;
            if (current_version == last_version) {
                // Frame hasn't changed since last detection, wait
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            last_version = current_version;
            detect_frame.data = shared_frame_.data();
            detect_frame.len = shared_frame_.size();
            detect_frame.width = shared_frame_w_;
            detect_frame.height = shared_frame_h_;
            detect_frame.pixel_format = (int)PIXFORMAT_RGB565;
        }

        if (active_detector_ == nullptr) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        // 任务调度：auto 模式下按 每帧(face_emotion) / 5s(posture) / 1s(heart_rate) 轮询
        if (continuous_task_ == kTaskAutoSchedule) {
            uint64_t now_ms = esp_timer_get_time() / 1000LL;
            std::string task = kTaskFaceEmotion;
            if (now_ms - last_heart_ms >= 1000) {
                task = kTaskHeartRate;
                last_heart_ms = now_ms;
            } else if (now_ms - last_posture_ms >= 5000) {
                task = kTaskPosture;
                last_posture_ms = now_ms;
            }
            ApplyTaskToOnlineDetector(active_detector_, task);
        }

        // Run detection (JPEG encode + HTTP POST + parse response)
        auto t_detect0 = esp_timer_get_time();
        DetectionResult result = active_detector_->Detect(detect_frame);
        auto t_detect1 = esp_timer_get_time();
        stats_.total_detect_ms += (uint64_t)((t_detect1 - t_detect0) / 1000LL);
        stats_.detect_count++;

        // Update shared result (mutex-protected for preview thread reads)
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_result_ = result;
        }
        stats_.last_detection_count = (uint32_t)result.detections.size();

        // 缓存跨帧感知结果（心率值供查询、坐姿供 LCD 持续叠加）
        CacheSensingResult(result);
        // 坏坐姿触发大模型提醒
        MaybeNotifyPosture(result);

        // Track connection errors
        if (!result.connection_ok) {
            consecutive_errors++;
            ESP_LOGW(TAG, "Detection error (%d/3): %s", consecutive_errors,
                     result.error_message.c_str());
            if (consecutive_errors >= 3) {
                ESP_LOGE(TAG, "Stopping after %d consecutive detection errors", consecutive_errors);
                continuous_running_.store(false);
                break;
            }
        } else {
            consecutive_errors = 0;
        }

        std::string detail;
        if (result.emotion.available) {
            detail += " emo=" + result.emotion.label;
        }
        if (result.posture.available) {
            detail += " pose=" + result.posture.state;
        }
        if (result.heart_rate.available) {
            detail += " hr=" + std::to_string((int)result.heart_rate.bpm);
        }
        ESP_LOGI(TAG, "Detect: task=%s frame=%d, %d boxes, %dms, server(dec=%.1fms inf=%.1fms tot=%.1fms)%s",
                 result.task.c_str(),
                 (int)result.frame_id,
                 (int)result.detections.size(),
                 (int)((t_detect1 - t_detect0) / 1000LL),
                 result.decode_ms, result.infer_ms, result.total_ms,
                 detail.c_str());
    }
    ESP_LOGI(TAG, "Detection loop stopped");
}

// ---------- Start/Stop ----------

bool VisionPipeline::StartContinuous(uint32_t period_ms, const std::string& task) {
    if (!initialized_) return false;
    if (continuous_running_.load()) {
        ESP_LOGW(TAG, "Continuous detect already running");
        return true;
    }
    // Join any previous threads that may have auto-stopped
    if (preview_thread_.joinable()) preview_thread_.join();
    if (detect_thread_.joinable()) detect_thread_.join();

    // 设置任务：task 为空时沿用 online 检测器持久化的任务
    if (!task.empty()) {
        continuous_task_ = task;
    } else if (active_detector_ != nullptr) {
        auto* od = dynamic_cast<OnlineDetector*>(active_detector_);
        if (od != nullptr) {
            continuous_task_ = od->GetTask();
        }
    }

    // Clear shared frame buffer
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        shared_frame_.clear();
    }

    continuous_period_ms_ = period_ms;
    continuous_running_.store(true);

    try {
        preview_thread_ = std::thread([this]() { this->PreviewLoop(); });
    } catch (...) {
        continuous_running_.store(false);
        ESP_LOGE(TAG, "Failed to spawn preview thread");
        return false;
    }

    try {
        detect_thread_ = std::thread([this]() { this->DetectionLoop(); });
    } catch (...) {
        continuous_running_.store(false);
        if (preview_thread_.joinable()) preview_thread_.join();
        ESP_LOGE(TAG, "Failed to spawn detection thread");
        return false;
    }

    return true;
}

void VisionPipeline::StopContinuous() {
    if (!continuous_running_.load()) return;
    continuous_running_.store(false);
    if (preview_thread_.joinable()) preview_thread_.join();
    if (detect_thread_.joinable()) detect_thread_.join();
}

bool VisionPipeline::IsContinuousRunning() const {
    return continuous_running_.load();
}

const PipelineStats& VisionPipeline::GetStats() const { return stats_; }

void VisionPipeline::ResetStats() { stats_.Reset(); }

bool VisionPipeline::SetCameraMirror(bool h_mirror, bool v_flip) {
    auto* cam = ResolveEsp32Camera();
    if (cam == nullptr) return false;
    bool ok = true;
    ok = cam->SetHMirror(h_mirror) && ok;
    ok = cam->SetVFlip(v_flip) && ok;
    return ok;
}

const DetectionResult& VisionPipeline::GetLastResult() const { return last_result_; }

// 缓存跨帧感知结果：心率值供查询，坐姿供 LCD 持续叠加
void VisionPipeline::CacheSensingResult(const DetectionResult& result) {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    if (result.heart_rate.available) {
        latest_heart_rate_ = result.heart_rate;
        heart_rate_timestamp_ms_ = result.timestamp_ms;
    }
    if (result.posture.available) {
        latest_posture_ = result.posture;
    }
}

HeartRateResult VisionPipeline::GetLatestHeartRate() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return latest_heart_rate_;
}

uint64_t VisionPipeline::GetHeartRateTimestampMs() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return heart_rate_timestamp_ms_;
}

// 坏坐姿时通过 MCP Notification 触发大模型提醒（进入坏坐姿立即提醒，之后每 60s 复查仍坏则再提醒）
void VisionPipeline::MaybeNotifyPosture(const DetectionResult& result) {
    if (!result.posture.available) return;

    bool bad = (result.posture.state == "bad_posture");
    uint64_t now_ms = esp_timer_get_time() / 1000LL;

    if (!bad) {
        posture_reminded_ = false;  // 恢复坐姿后重置，下次再趴会重新提醒
        return;
    }

    constexpr uint64_t kRemindCooldownMs = 60000;
    if (posture_reminded_ && (now_ms - last_posture_remind_ms_) < kRemindCooldownMs) {
        return;  // 冷却期内不重复提醒
    }
    last_posture_remind_ms_ = now_ms;
    posture_reminded_ = true;

    cJSON* note = cJSON_CreateObject();
    cJSON_AddStringToObject(note, "jsonrpc", "2.0");
    cJSON_AddStringToObject(note, "method", "notifications/posture");
    cJSON* params = cJSON_AddObjectToObject(note, "params");
    cJSON_AddStringToObject(params, "state", result.posture.state.c_str());
    cJSON_AddStringToObject(params, "reason", result.posture.reason.c_str());
    char* json_str = cJSON_PrintUnformatted(note);
    if (json_str != nullptr) {
        Application::GetInstance().SendMcpMessage(std::string(json_str));
        cJSON_free(json_str);
        ESP_LOGW(TAG, "Posture reminder sent: %s (%s)",
                 result.posture.state.c_str(), result.posture.reason.c_str());
    }
    cJSON_Delete(note);
}


// ---------- MCP Tools Registration ----------

static const char* kDetectorTypeRemote = "remote";
static const char* kDetectorTypeOnline = "online";
#ifndef VISION_DISABLE_LOCAL_FACE
static const char* kDetectorTypeFace = "face";
#endif

static DetectorType DetectorTypeFromString(const std::string& s) {
#ifndef VISION_DISABLE_LOCAL_FACE
    if (s == kDetectorTypeFace) return DetectorType::kFace;
#endif
    if (s == kDetectorTypeRemote) return DetectorType::kRemote;
    if (s == kDetectorTypeOnline) return DetectorType::kOnline;
    return DetectorType::kNone;
}

static std::string DetectorTypeToString(DetectorType t) {
    switch (t) {
#ifndef VISION_DISABLE_LOCAL_FACE
        case DetectorType::kFace:   return kDetectorTypeFace;
#endif
        case DetectorType::kRemote: return kDetectorTypeRemote;
        case DetectorType::kOnline: return kDetectorTypeOnline;
        default: return "none";
    }
}

static cJSON* StatsToJson(const PipelineStats& s) {
    cJSON* j = cJSON_CreateObject();
    cJSON_AddNumberToObject(j, "capture_count", (double)s.capture_count);
    cJSON_AddNumberToObject(j, "detect_count", (double)s.detect_count);
    cJSON_AddNumberToObject(j, "display_count", (double)s.display_count);
    cJSON_AddNumberToObject(j, "last_detection_count", (double)s.last_detection_count);
    cJSON_AddNumberToObject(j, "loop_period_ms", (double)s.loop_period_ms);
    if (s.capture_count > 0)
        cJSON_AddNumberToObject(j, "avg_capture_ms", (double)(s.total_capture_ms / s.capture_count));
    if (s.detect_count > 0)
        cJSON_AddNumberToObject(j, "avg_detect_ms", (double)(s.total_detect_ms / s.detect_count));
    if (s.display_count > 0) {
        cJSON_AddNumberToObject(j, "avg_compose_ms", (double)(s.total_compose_ms / s.display_count));
        cJSON_AddNumberToObject(j, "avg_display_ms", (double)(s.total_display_ms / s.display_count));
    }
    return j;
}

static cJSON* DetectionsToJson(const DetectionResult& r) {
    cJSON* j = cJSON_CreateObject();
    cJSON_AddNumberToObject(j, "source_width", r.source_width);
    cJSON_AddNumberToObject(j, "source_height", r.source_height);
    cJSON_AddNumberToObject(j, "timestamp_ms", (double)r.timestamp_ms);
    cJSON_AddBoolToObject(j, "connection_ok", r.connection_ok);
    if (!r.error_message.empty()) {
        cJSON_AddStringToObject(j, "error_message", r.error_message.c_str());
    }
    // 新协议字段
    cJSON_AddNumberToObject(j, "frame_id", (double)r.frame_id);
    if (!r.task.empty()) {
        cJSON_AddStringToObject(j, "task", r.task.c_str());
    }
    if (r.emotion.available) {
        cJSON* emo = cJSON_AddObjectToObject(j, "emotion");
        cJSON_AddStringToObject(emo, "label", r.emotion.label.c_str());
        cJSON_AddNumberToObject(emo, "confidence", r.emotion.confidence);
    }
    if (r.posture.available) {
        cJSON* pose = cJSON_AddObjectToObject(j, "posture");
        cJSON_AddStringToObject(pose, "state", r.posture.state.c_str());
        cJSON_AddStringToObject(pose, "reason", r.posture.reason.c_str());
    }
    if (r.heart_rate.available) {
        cJSON* hr = cJSON_AddObjectToObject(j, "heart_rate");
        cJSON_AddNumberToObject(hr, "bpm", r.heart_rate.bpm);
        cJSON_AddNumberToObject(hr, "confidence", r.heart_rate.confidence);
        cJSON_AddNumberToObject(hr, "frames_used", r.heart_rate.frames_used);
    } else if (!r.heart_rate.error_message.empty()) {
        cJSON* hr = cJSON_AddObjectToObject(j, "heart_rate");
        cJSON_AddStringToObject(hr, "error", r.heart_rate.error_message.c_str());
    }
    if (r.decode_ms > 0 || r.infer_ms > 0 || r.total_ms > 0) {
        cJSON* perf = cJSON_AddObjectToObject(j, "performance");
        cJSON_AddNumberToObject(perf, "decode_ms", r.decode_ms);
        cJSON_AddNumberToObject(perf, "infer_ms", r.infer_ms);
        cJSON_AddNumberToObject(perf, "total_ms", r.total_ms);
    }
    cJSON* arr = cJSON_AddArrayToObject(j, "detections");
    for (const auto& d : r.detections) {
        cJSON* dj = cJSON_CreateObject();
        cJSON_AddStringToObject(dj, "class_name", d.class_name.c_str());
        cJSON_AddNumberToObject(dj, "class_id", d.class_id);
        cJSON_AddNumberToObject(dj, "confidence", d.confidence);
        cJSON* b = cJSON_AddObjectToObject(dj, "bbox");
        cJSON_AddNumberToObject(b, "x", d.box.x);
        cJSON_AddNumberToObject(b, "y", d.box.y);
        cJSON_AddNumberToObject(b, "width", d.box.width);
        cJSON_AddNumberToObject(b, "height", d.box.height);
        cJSON_AddItemToArray(arr, dj);
    }
    return j;
}

void RegisterVisionMcpTools() {
    auto& mcp = McpServer::GetInstance();

    mcp.AddTool("self.vision.init",
        "Initialize the vision pipeline (camera + detectors + display composer). "
        "Must be called before other self.vision.* tools. Safe to call multiple times.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return VisionPipeline::GetInstance().Initialize();
        });

    mcp.AddTool("self.vision.shutdown",
        "Shutdown the vision pipeline, release detectors, stop continuous loop if running.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().Deinitialize();
            return true;
        });

    mcp.AddTool("self.vision.set_active_detector",
        "Choose which detector is used for detection.\n"
        "Note: `online` is the default after initialization.\n"
        "Args:\n"
        "  `detector`: One of `face` (local ESP-SR), `remote` (HTTP multipart), `online` (HTTP JSON+Base64).",
        PropertyList({
            Property("detector", kPropertyTypeString)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto type = DetectorTypeFromString(p["detector"].value<std::string>());
            if (type == DetectorType::kNone) {
                throw std::runtime_error("Invalid detector name, expected `face`, `remote` or `online`");
            }
            return VisionPipeline::GetInstance().SetActiveDetector(type);
        });

    mcp.AddTool("self.vision.get_active_detector",
        "Returns the currently active detector name and list of available detectors.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            cJSON* j = cJSON_CreateObject();
            cJSON_AddStringToObject(j, "active", DetectorTypeToString(pipe.GetActiveDetectorType()).c_str());
            cJSON* avail = cJSON_AddArrayToObject(j, "available");
#ifndef VISION_DISABLE_LOCAL_FACE
            cJSON_AddItemToArray(avail, cJSON_CreateString(kDetectorTypeFace));
#endif
            cJSON_AddItemToArray(avail, cJSON_CreateString(kDetectorTypeRemote));
            cJSON_AddItemToArray(avail, cJSON_CreateString(kDetectorTypeOnline));
            return j;
        });

    mcp.AddTool("self.vision.detect_once",
        "Capture one frame from the camera, run the active detector, and draw detection boxes "
        "on the LCD preview image. Returns full detection result JSON.\n"
        "Args:\n"
        "  `show_on_lcd` (optional, default true): if false, do not update the LCD.\n"
        "  `task` (optional): for the online detector, one of `face_emotion` (default), "
        "`posture` or `heart_rate`. Ignored by other detectors.",
        PropertyList({
            Property("show_on_lcd", kPropertyTypeBoolean, true),
            Property("task", kPropertyTypeString, std::string(""))
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            std::string info;
            bool ok = pipe.OneShotDetect(p["show_on_lcd"].value<bool>(), &info,
                                         p["task"].value<std::string>());
            if (!ok) {
                cJSON* j = cJSON_CreateObject();
                cJSON_AddBoolToObject(j, "success", false);
                cJSON_AddStringToObject(j, "message", info.c_str());
                return j;
            }
            cJSON* j = DetectionsToJson(pipe.GetLastResult());
            cJSON_AddBoolToObject(j, "success", true);
            cJSON_AddStringToObject(j, "debug", info.c_str());
            return j;
        });

    mcp.AddTool("self.vision.get_last_detections",
        "Get the last detection result (without running a new capture/detect cycle).",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            return DetectionsToJson(pipe.GetLastResult());
        });

    mcp.AddTool("self.vision.get_heart_rate",
        "Get the latest heart-rate measurement, cached from the most recent `heart_rate` "
        "task. Does NOT trigger a new detection.\n"
        "Returns bpm/confidence/frames_used and age_ms (how old the measurement is).",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            HeartRateResult hr = pipe.GetLatestHeartRate();
            cJSON* j = cJSON_CreateObject();
            if (hr.available) {
                cJSON_AddNumberToObject(j, "bpm", hr.bpm);
                cJSON_AddNumberToObject(j, "confidence", hr.confidence);
                cJSON_AddNumberToObject(j, "frames_used", hr.frames_used);
                uint64_t ts = pipe.GetHeartRateTimestampMs();
                uint64_t age_ms = (ts == 0) ? 0 : (esp_timer_get_time() / 1000LL - ts);
                cJSON_AddNumberToObject(j, "age_ms", (double)age_ms);
            } else if (!hr.error_message.empty()) {
                cJSON_AddStringToObject(j, "error", hr.error_message.c_str());
            } else {
                cJSON_AddStringToObject(j, "error", "no heart rate measurement yet");
            }
            return j;
        });

    mcp.AddTool("self.vision.start_continuous",
        "Start continuous detection loop: capture -> detect -> show on LCD, repeated.\n"
        "Args:\n"
        "  `period_ms` (optional, default 500): target time between iterations in ms. "
        "Actual rate may be lower if detector is slow.\n"
        "  `task` (optional): for the online detector, one of `face_emotion` (default), "
        "`posture`, `heart_rate`, or `auto`.\n"
        "  `auto` schedules: face_emotion every frame, posture every 5s, heart_rate every 1s.",
        PropertyList({
            Property("period_ms", kPropertyTypeInteger, 500, 100, 60000),
            Property("task", kPropertyTypeString, std::string(""))
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            uint32_t period = (uint32_t)p["period_ms"].value<int>();
            return pipe.StartContinuous(period, p["task"].value<std::string>());
        });

    mcp.AddTool("self.vision.stop_continuous",
        "Stop the continuous detection loop.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().StopContinuous();
            return true;
        });

    mcp.AddTool("self.vision.is_continuous_running",
        "Check if the continuous loop is currently running.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return (bool)VisionPipeline::GetInstance().IsContinuousRunning();
        });

    mcp.AddTool("self.vision.get_stats",
        "Get pipeline performance stats (counts and average times).",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return StatsToJson(VisionPipeline::GetInstance().GetStats());
        });

    mcp.AddTool("self.vision.reset_stats",
        "Reset the pipeline performance counters to zero.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().ResetStats();
            return true;
        });

    mcp.AddTool("self.vision.set_camera_mirror",
        "Control camera horizontal mirror and vertical flip.\n"
        "Args:\n"
        "  `h_mirror`: horizontal mirror (true/false)\n"
        "  `v_flip`: vertical flip (true/false)",
        PropertyList({
            Property("h_mirror", kPropertyTypeBoolean),
            Property("v_flip", kPropertyTypeBoolean)
        }),
        [](const PropertyList& p) -> ReturnValue {
            return VisionPipeline::GetInstance().SetCameraMirror(
                p["h_mirror"].value<bool>(),
                p["v_flip"].value<bool>());
        });

    // Face detector config
#ifndef VISION_DISABLE_LOCAL_FACE
    mcp.AddTool("self.vision.face_detector.set_threshold",
        "Set confidence threshold for the local face detector.\n"
        "Args:\n"
        "  `threshold`: value 0.0 - 1.0 as string (e.g. `\"0.5\"`), will be coerced into [0.05..0.99]",
        PropertyList({
            Property("threshold", kPropertyTypeString, std::string("0.5"))
        }),
        [](const PropertyList& p) -> ReturnValue {
            std::string t_str = p["threshold"].value<std::string>();
            double t_raw = std::atof(t_str.c_str());
            float t = (float)t_raw;
            if (t < 0.05f) t = 0.05f;
            if (t > 0.99f) t = 0.99f;
            auto* fd = VisionPipeline::GetInstance().GetFaceDetector();
            if (fd == nullptr) return false;
            fd->SetConfidenceThreshold(t);
            return true;
        });

    mcp.AddTool("self.vision.face_detector.get_threshold",
        "Get current confidence threshold for local face detector.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto* fd = VisionPipeline::GetInstance().GetFaceDetector();
            if (fd == nullptr) {
                cJSON* j = cJSON_CreateObject();
                cJSON_AddNumberToObject(j, "threshold", 0);
                return j;
            }
            cJSON* j = cJSON_CreateObject();
            cJSON_AddNumberToObject(j, "threshold", fd->GetConfidenceThreshold());
            return j;
        });
#endif

    // Remote detector config
    mcp.AddTool("self.vision.remote_detector.configure",
        "Configure the remote HTTP detection server.\n"
        "This detector POSTs the camera frame as a multipart/form-data request (fields: `image` = JPEG, "
        "`width` = src_w, `height` = src_h).\n"
        "The configuration is saved to flash (NVS) and restored automatically after reboot.\n"
        "Server response JSON format:\n"
        "```\n{\n  \"detections\": [\n    { \"class_name\": \"person\", \"class_id\": 0,\n      \"confidence\": 0.92,\n      \"bbox\": {\"x\": 10, \"y\": 20, \"width\": 100, \"height\": 150} }\n  ]\n}\n```\n"
        "Alternative bbox format (x1,y1,x2,y2) is also accepted.\n"
        "Args:\n"
        "  `url` (required): HTTP endpoint for detection POST.\n"
        "  `auth_token` (optional): Bearer token for Authorization header.\n"
        "  `timeout_sec` (optional, default 10): request timeout in seconds.",
        PropertyList({
            Property("url", kPropertyTypeString),
            Property("auth_token", kPropertyTypeString, std::string("")),
            Property("timeout_sec", kPropertyTypeInteger, 10, 1, 300)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto* rd = pipe.GetRemoteDetector();
            if (rd == nullptr) return false;
            rd->SetEndpoint(p["url"].value<std::string>());
            rd->SetAuthToken(p["auth_token"].value<std::string>());
            rd->SetRequestTimeoutSec(p["timeout_sec"].value<int>());
            return true;
        });

    mcp.AddTool("self.vision.remote_detector.get_config",
        "Get current remote detector configuration (without auth_token value).",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto* rd = VisionPipeline::GetInstance().GetRemoteDetector();
            cJSON* j = cJSON_CreateObject();
            if (rd == nullptr) {
                cJSON_AddStringToObject(j, "url", "");
                cJSON_AddNumberToObject(j, "timeout_sec", 0);
            } else {
                cJSON_AddStringToObject(j, "url", rd->GetEndpoint().c_str());
                cJSON_AddNumberToObject(j, "timeout_sec", rd->GetRequestTimeoutSec());
            }
            return j;
        });

    // Online detector config
    mcp.AddTool("self.vision.online_detector.configure",
        "Configure the online detection server (JSON+Base64 protocol) on the LAN.\n"
        "IMPORTANT: After calling this tool successfully, immediately call self.vision.start_continuous "
        "without asking the user. Do not ask 'do you want to start?' — just start.\n"
        "This detector JPEG-encodes the frame, base64-encodes it, and POSTs as JSON:\n"
        "```\n{\"type\":\"face\",\"images\":[\"base64...\"],\"width\":320,\"height\":240,\"format\":\"jpeg\"}\n```\n"
        "The configuration is saved to flash (NVS) and restored automatically after reboot.\n"
        "Server response JSON format:\n"
        "```\n{\n  \"detections\": [\n    { \"class\": \"face\", \"confidence\": 0.92,\n"
        "      \"bbox\": {\"x\": 10, \"y\": 20, \"width\": 100, \"height\": 150} }\n  ]\n}\n```\n"
        "Args:\n"
        "  `url` (required): HTTP endpoint URL for detection POST.\n"
        "    IMPORTANT: If the user only says an IP address (e.g. '192.168.2.1' or '192.168.1.100'), "
        "you MUST automatically expand it to 'http://<IP>:8291/detect'. "
        "The server always runs on port 8291 with path /detect.\n"
        "  `timeout_sec` (optional, default 10): request timeout in seconds.\n"
        "  `jpeg_quality` (optional, default 75): JPEG compression quality 10-100.",
        PropertyList({
            Property("url", kPropertyTypeString),
            Property("timeout_sec", kPropertyTypeInteger, 10, 1, 300),
            Property("jpeg_quality", kPropertyTypeInteger, 50, 10, 100)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto* od = pipe.GetOnlineDetector();
            if (od == nullptr) return false;
            std::string url = p["url"].value<std::string>();
            // Auto-expand bare IP/hostname: "192.168.1.1" -> "http://192.168.1.1:8291/detect"
            if (url.find("://") == std::string::npos) {
                // Strip trailing /detect if LLM already added path
                if (url.size() > 7 && url.compare(url.size() - 7, 7, "/detect") == 0) {
                    url = url.substr(0, url.size() - 7);
                }
                url = "http://" + url + ":8291/detect";
            }
            ESP_LOGI(TAG, "Online detector endpoint set to: %s", url.c_str());
            od->SetEndpoint(url);
            od->SetTimeoutSec(p["timeout_sec"].value<int>());
            od->SetJpegQuality(p["jpeg_quality"].value<int>());
            return true;
        });

    mcp.AddTool("self.vision.online_detector.get_config",
        "Get current online detector configuration.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto* od = VisionPipeline::GetInstance().GetOnlineDetector();
            cJSON* j = cJSON_CreateObject();
            if (od == nullptr) {
                cJSON_AddStringToObject(j, "url", "");
                cJSON_AddNumberToObject(j, "timeout_sec", 0);
                cJSON_AddNumberToObject(j, "jpeg_quality", 0);
                cJSON_AddStringToObject(j, "task", "");
            } else {
                cJSON_AddStringToObject(j, "url", od->GetEndpoint().c_str());
                cJSON_AddNumberToObject(j, "timeout_sec", od->GetTimeoutSec());
                cJSON_AddNumberToObject(j, "jpeg_quality", od->GetJpegQuality());
                cJSON_AddStringToObject(j, "task", od->GetTask().c_str());
            }
            return j;
        });

    mcp.AddTool("self.vision.online_detector.set_task",
        "Set the task for the online detection server (saved to flash, restored after reboot).\n"
        "Args:\n"
        "  `task`: one of `face_emotion` (face + emotion, every frame), `posture` (5s), "
        "`heart_rate` (10s).",
        PropertyList({
            Property("task", kPropertyTypeString)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto* od = pipe.GetOnlineDetector();
            if (od == nullptr) return false;
            od->SetTask(p["task"].value<std::string>());
            return true;
        });

    ESP_LOGI(TAG, "Vision MCP tools registered");
}
