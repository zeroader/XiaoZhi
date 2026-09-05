#include "vision_pipeline.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <esp_pthread.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <cJSON.h>

#include "board.h"
#include "esp32_camera.h"
#include "lvgl_display.h"
#include "mcp_server.h"
#include "system_info.h"
#include "application.h"
#include "assets.h"
#include "settings.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "esp_jpeg_dec.h"

#define TAG "VisionPipeline"

static void StopQueuedVoiceAlert();

// Detection and lamp endpoints share one Flask server. Keep the lamp endpoint
// in sync whenever a vision server URL is configured.
static void SyncLampServerUrl(const std::string& endpoint) {
    std::string base = endpoint;
    const auto scheme = base.find("://");
    const auto path = base.find('/', scheme == std::string::npos ? 0 : scheme + 3);
    if (path != std::string::npos) {
        base.resize(path);
    }
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    if (!base.empty()) {
        Settings settings("lamp", true);
        settings.SetString("server_url", base);
        ESP_LOGI(TAG, "Lamp server URL synchronized to: %s", base.c_str());
    }
}

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
    if (display == nullptr) {
        ESP_LOGW(TAG, "ResolveLvglDisplay: Board::GetDisplay() returns NULL");
        return nullptr;
    }
    auto* lcd = dynamic_cast<LvglDisplay*>(display);
    if (lcd == nullptr) {
        ESP_LOGW(TAG, "ResolveLvglDisplay: dynamic_cast<LvglDisplay*> failed, display is not an LvglDisplay subclass");
    }
    return lcd;
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

    online_detector_ = std::make_unique<OnlineDetector>();
    if (!online_detector_->Initialize()) {
        ESP_LOGW(TAG, "Online detector initialize returned false");
    }

    display_composer_ = std::make_unique<VisionDisplay>();

    // Restore the detector selected by the user; online remains the default.
    // Migrate the legacy remote mode so stale NVS values cannot trigger
    // requests to an obsolete detection server.
    active_detector_type_ = DetectorType::kOnline;
    active_detector_ = online_detector_.get();
    {
        Settings settings("vision");
        const auto saved = settings.GetString("active_detector");
        if (saved == "remote") {
            Settings migrated("vision", true);
            migrated.SetString("active_detector", "online");
            migrated.EraseKey("remote_url");
            migrated.EraseKey("remote_token");
            migrated.EraseKey("remote_timeout");
            ESP_LOGW(TAG, "Migrated legacy remote detector configuration to online");
        }
#ifndef VISION_DISABLE_LOCAL_FACE
        else if (saved == "face" && face_detector_ != nullptr) {
            active_detector_type_ = DetectorType::kFace;
            active_detector_ = face_detector_.get();
        }
#endif
    }

    initialized_ = true;
    ESP_LOGI(TAG, "Vision pipeline initialized (active=%s, camera=%s)",
             active_detector_->GetName(),
             has_camera_ ? "ok" : "none");
    return true;
}

void VisionPipeline::Deinitialize() {
    StopQueuedVoiceAlert();
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

    // 释放 JPEG 解码器与预览缓冲（OV2640 直出 JPEG 时使用）
    if (jpeg_dec_ != nullptr) {
        jpeg_dec_close((jpeg_dec_handle_t)jpeg_dec_);
        jpeg_dec_ = nullptr;
    }
    if (preview_rgb565_ != nullptr) {
        heap_caps_free(preview_rgb565_);
        preview_rgb565_ = nullptr;
        preview_rgb565_cap_ = 0;
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
        case DetectorType::kRemote: return nullptr;
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

static bool ResultIncludesTask(const std::string& task_list, const char* task) {
    if (task == nullptr || *task == '\0') return false;
    size_t start = 0;
    while (start < task_list.size()) {
        const size_t end = task_list.find(',', start);
        const size_t length = (end == std::string::npos ? task_list.size() : end) - start;
        if (length == strlen(task) && task_list.compare(start, length, task) == 0) {
            return true;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

bool VisionPipeline::SetActiveDetector(DetectorType type) {
    if (type == DetectorType::kRemote) {
        ESP_LOGW(TAG, "Legacy remote detector is disabled; use online detector");
        return false;
    }
    Detector* d = GetDetector(type);
    if (d == nullptr) {
        ESP_LOGE(TAG, "Invalid detector type: %d", (int)type);
        return false;
    }
    active_detector_type_ = type;
    active_detector_ = d;
    Settings settings("vision", true);
    const char* detector_name = type == DetectorType::kFace ? "face"
        : (type == DetectorType::kRemote ? "remote" : "online");
    settings.SetString("active_detector", detector_name);
    ESP_LOGI(TAG, "Active detector changed to: %s", d->GetName());
    return true;
}

#ifndef VISION_DISABLE_LOCAL_FACE
FaceDetector* VisionPipeline::GetFaceDetector() { return face_detector_.get(); }
#endif
RemoteDetector* VisionPipeline::GetRemoteDetector() { return nullptr; }
OnlineDetector* VisionPipeline::GetOnlineDetector() { return online_detector_.get(); }
VisionDisplay* VisionPipeline::GetDisplayComposer() { return display_composer_.get(); }

bool VisionPipeline::CaptureFrame(ImageFrame& out_frame) {
    auto* cam = ResolveEsp32Camera();
    if (cam == nullptr) {
        ESP_LOGE(TAG, "CaptureFrame: No Esp32Camera available for capture. "
                 "Check that Board::GetCamera() returns non-NULL (camera may not be plugged in, "
                 "or XL9555 PWDN/RESET pins are not set correctly)");
        return false;
    }
    auto t0 = esp_timer_get_time();
    if (!cam->Capture()) {
        ESP_LOGE(TAG, "CaptureFrame: Camera Capture() returned false. Check camera cable / PWDN pin.");
        return false;
    }
    auto t1 = esp_timer_get_time();
    stats_.total_capture_ms += (uint64_t)((t1 - t0) / 1000LL);
    stats_.capture_count++;

    camera_fb_t* fb = cam->GetLastCapturedFrame();
    if (fb == nullptr) {
        ESP_LOGE(TAG, "CaptureFrame: GetLastCapturedFrame() returned NULL frame buffer");
        return false;
    }
    if (fb->buf == nullptr || fb->width == 0 || fb->height == 0) {
        ESP_LOGE(TAG, "CaptureFrame: invalid frame buf=%p %dx%d len=%u",
                 fb->buf, fb->width, fb->height, (unsigned)fb->len);
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

// 将 OV2640 直出的 JPEG 帧解码为 RGB565（esp_new_jpeg 软解码），仅用于 LCD 预览。
// 检测线程无需解码——JPEG 字节直接上传到服务器，编码成本为零。
bool VisionPipeline::DecodeJpegFrame(const ImageFrame& jpeg, uint8_t** out_rgb565,
                                     size_t* out_len, int* out_w, int* out_h) {
    if (jpeg.data == nullptr || jpeg.len == 0) return false;

    // 懒初始化解码器（仅预览线程使用，无并发访问）
    if (jpeg_dec_ == nullptr) {
        jpeg_dec_config_t config = {};
        // ComposePreview 会将输入 RGB565 转为 LVGL 所需的字节序，
        // 因此这里保持与 OV2640 RGB565 原始帧一致的 big-endian 格式。
        config.output_type = JPEG_PIXEL_FORMAT_RGB565_BE;
        config.rotate = JPEG_ROTATE_0D;
        if (jpeg_dec_open(&config, &jpeg_dec_) != JPEG_ERR_OK) {
            ESP_LOGE(TAG, "DecodeJpegFrame: failed to open JPEG decoder");
            jpeg_dec_ = nullptr;
            return false;
        }
    }

    jpeg_dec_io_t io = {};
    io.inbuf = (uint8_t*)jpeg.data;
    io.inbuf_len = (int)jpeg.len;

    jpeg_dec_header_info_t info = {};
    if (jpeg_dec_parse_header(jpeg_dec_, &io, &info) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "DecodeJpegFrame: parse header failed");
        return false;
    }

    size_t need = (size_t)info.width * info.height * 2;  // RGB565
    if (need == 0 || need > 1024 * 1024) {
        ESP_LOGE(TAG, "DecodeJpegFrame: bad dims %dx%d", info.width, info.height);
        return false;
    }
    if (preview_rgb565_ == nullptr || preview_rgb565_cap_ < need) {
        if (preview_rgb565_ != nullptr) heap_caps_free(preview_rgb565_);
        // esp_new_jpeg 要求 S3 上输出缓冲 16 字节对齐
        preview_rgb565_ = (uint8_t*)heap_caps_aligned_alloc(16, need, MALLOC_CAP_SPIRAM);
        preview_rgb565_cap_ = preview_rgb565_ != nullptr ? need : 0;
    }
    if (preview_rgb565_ == nullptr) {
        ESP_LOGE(TAG, "DecodeJpegFrame: no memory for %d bytes", (int)need);
        return false;
    }

    io.outbuf = preview_rgb565_;
    // jpeg_dec_parse_header 已在 io 中更新输入位置；不要再次手动偏移，
    // 否则会跳过 JPEG 数据而导致解码画面损坏。
    if (jpeg_dec_process(jpeg_dec_, &io) != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "DecodeJpegFrame: decode failed");
        return false;
    }

    *out_rgb565 = preview_rgb565_;
    *out_len = need;
    *out_w = info.width;
    *out_h = info.height;
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

    // 连续检测运行时摄像头被 PreviewLoop 独占，此处不再取帧，
    // 避免并发 esp_camera_fb_get 失败被大模型误判为"摄像头坏了"。
    // 引导大模型改用缓存读取工具。
    if (continuous_running_.load() || preview_running_.load()) {
        if (out_debug_info) {
            *out_debug_info = "continuous detection is running (camera busy). "
                              "Use self.vision.get_user_emotion / get_posture / "
                              "get_heart_rate / get_last_detections instead";
        }
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
    MaybePlayVoiceAlerts(result);

    bool displayed = false;
    if (show_on_lcd && display_composer_) {
        auto* lcd = ResolveLvglDisplay();
        if (lcd == nullptr) {
            ESP_LOGW(TAG, "OneShotDetect: ResolveLvglDisplay returned NULL, skip showing on LCD");
        } else {
            auto t_comp0 = esp_timer_get_time();
            ESP_LOGI(TAG, "OneShotDetect: ComposePreview frame=%dx%d format=%d len=%u",
                     frame.width, frame.height, frame.pixel_format, (unsigned)frame.len);
            // OV2640 直出 JPEG：解码成 RGB565 再送 LCD
            const uint8_t* disp_data = frame.data;
            int disp_w = frame.width;
            int disp_h = frame.height;
            if (frame.pixel_format == PIXFORMAT_JPEG) {
                uint8_t* dec = nullptr; size_t dec_len = 0; int dw = 0, dh = 0;
                if (DecodeJpegFrame(frame, &dec, &dec_len, &dw, &dh)) {
                    disp_data = dec; disp_w = dw; disp_h = dh;
                }
            }
            auto image = display_composer_->ComposePreview(disp_data, disp_w, disp_h,
                                                          (size_t)disp_w * 2, result);
            auto t_comp1 = esp_timer_get_time();
            stats_.total_compose_ms += (uint64_t)((t_comp1 - t_comp0) / 1000LL);

            if (!image) {
                ESP_LOGW(TAG, "OneShotDetect: ComposePreview returned NULL (bad alloc or invalid dims?)");
            } else {
                auto t_disp0 = esp_timer_get_time();
                lcd->SetVisionPreviewImage(std::move(image), result);
                auto t_disp1 = esp_timer_get_time();
                stats_.total_display_ms += (uint64_t)((t_disp1 - t_disp0) / 1000LL);
                stats_.display_count++;
                displayed = true;
                ESP_LOGI(TAG, "OneShotDetect: SetPreviewImage done (took %dms)",
                         (int)((t_disp1 - t_disp0) / 1000LL));
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
    ESP_LOGI(TAG, "Preview loop started (15 FPS capture, 12 FPS LCD)");

    constexpr uint32_t kPreviewPeriodMs = 67;  // ~15 FPS
    constexpr uint32_t kDisplayPeriodMs = 83;  // 12 FPS target; audio remains higher priority.
    uint64_t last_iter = 0;
    uint64_t last_display_ms = 0;

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
                shared_frame_pixfmt_ = frame.pixel_format;
                shared_frame_version_++;
            }

            // Release the camera buffer before doing any JPEG/LVGL work.  The
            // camera often has only one framebuffer, so holding it during a
            // display refresh stalls the next capture.
            const uint64_t now_ms = esp_timer_get_time() / 1000LL;
            const bool should_display = last_display_ms == 0 ||
                                        now_ms - last_display_ms >= kDisplayPeriodMs;
            ImageFrame display_frame = frame;
            if (should_display) {
                display_frame_.assign(frame.data, frame.data + frame.len);
            }
            ReleaseCurrentFrame();

            // Step 3: Compose preview with latest bbox (if any), at a bounded
            // LCD rate. Detection continues to receive every captured frame.
            if (should_display && display_composer_) {
                last_display_ms = now_ms;
                auto* lcd = ResolveLvglDisplay();
                if (lcd != nullptr) {
                    DetectionResult current_result;
                    {
                        std::lock_guard<std::mutex> lock(result_mutex_);
                        current_result = last_result_;
                    }
                    // 合并跨帧缓存的坐姿/心率/血压，保证 LCD 上持续显示（不被每帧 face_emotion 覆盖）
                    {
                        std::lock_guard<std::mutex> lock(sensing_mutex_);
                        current_result.emotion = latest_emotion_;
                        current_result.posture = latest_posture_;
                        current_result.heart_rate = latest_heart_rate_;
                        current_result.blood_pressure = latest_blood_pressure_;
                    }
                    // OV2640 直出 JPEG：解码成 RGB565 再送 LCD（检测线程直接用 JPEG 上传，互不影响）
                    display_frame.data = display_frame_.data();
                    const uint8_t* disp_data = display_frame.data;
                    int disp_w = display_frame.width;
                    int disp_h = display_frame.height;
                    if (display_frame.pixel_format == PIXFORMAT_JPEG) {
                        uint8_t* dec = nullptr; size_t dec_len = 0; int dw = 0, dh = 0;
                        if (!DecodeJpegFrame(display_frame, &dec, &dec_len, &dw, &dh)) {
                            disp_data = nullptr;  // 解码失败则本次不显示
                        } else {
                            disp_data = dec; disp_w = dw; disp_h = dh;
                        }
                    }
                    if (disp_data != nullptr) {
                        auto t_comp0 = esp_timer_get_time();
                        auto image = display_composer_->ComposePreview(
                            disp_data, disp_w, disp_h,
                            (size_t)disp_w * 2, current_result);
                        auto t_comp1 = esp_timer_get_time();
                        stats_.total_compose_ms += (uint64_t)((t_comp1 - t_comp0) / 1000LL);

                        if (image) {
                            auto t_disp0 = esp_timer_get_time();
                            lcd->SetVisionPreviewImage(std::move(image), current_result);
                            auto t_disp1 = esp_timer_get_time();
                            stats_.total_display_ms += (uint64_t)((t_disp1 - t_disp0) / 1000LL);
                            stats_.display_count++;
                        }
                    }
                }
            }

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
    uint64_t last_heart_ms = 0;    // 距上次 heart_rate 任务的毫秒数
    uint64_t last_blood_pressure_ms = 0;  // auto 模式下每约 2s 查询一次 BP 状态
    uint64_t retry_after_ms = 0;

    while (continuous_running_.load()) {
        // Wait for a NEW shared frame (skip if same as last processed)
        ImageFrame detect_frame;
        // 检测（HTTP 上传）在线程锁外执行，因此必须拥有独立的帧副本。
        // 不能只保存 shared_frame_.data()：预览线程可能在上传期间用下一帧
        // 覆盖或重分配 shared_frame_，从而发送损坏的 JPEG 并触发服务器 400。
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
            detect_frame_ = shared_frame_;
            detect_frame.data = detect_frame_.data();
            detect_frame.len = detect_frame_.size();
            detect_frame.width = shared_frame_w_;
            detect_frame.height = shared_frame_h_;
            detect_frame.pixel_format = shared_frame_pixfmt_;  // JPEG 直出时上传零编码
        }

        if (active_detector_ == nullptr) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        const uint64_t now_ms = esp_timer_get_time() / 1000LL;
        if (now_ms < retry_after_ms) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        // 自动模式每帧只上传一次 JPEG：服务器顺序执行情绪和坐姿，并复用
        // 人脸检测结果；心率/血压到期时加入同一请求，避免额外 HTTP 连接。
        std::vector<std::string> tasks;
        if (continuous_task_ == kTaskAutoSchedule) {
            tasks = {kTaskFaceEmotion, kTaskPosture};
            if (now_ms - last_blood_pressure_ms >= 2000) {
                tasks.push_back(kTaskBloodPressure);
                last_blood_pressure_ms = now_ms;
            }
            if (now_ms - last_heart_ms >= 10000) {
                tasks.push_back(kTaskHeartRate);
                last_heart_ms = now_ms;
            }
        } else {
            tasks.push_back(continuous_task_);
        }

        // Run detection (JPEG encode + HTTP POST + parse response)
        auto t_detect0 = esp_timer_get_time();
        DetectionResult result;
        if (auto* online = dynamic_cast<OnlineDetector*>(active_detector_)) {
            result = online->Detect(detect_frame, tasks);
        } else {
            // 本地检测器没有批量网络协议，保持其单任务行为。
            result = active_detector_->Detect(detect_frame);
        }
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
        // 根据服务器结果触发本地语音提醒
        MaybePlayVoiceAlerts(result);

        // Track connection errors
        if (!result.connection_ok) {
            consecutive_errors++;
            ESP_LOGW(TAG, "Detection error (%d/3): %s", consecutive_errors,
                     result.error_message.c_str());
            if (consecutive_errors >= 3) {
                ESP_LOGW(TAG, "Detection temporarily backing off after %d consecutive errors",
                         consecutive_errors);
                vTaskDelay(pdMS_TO_TICKS(1000));
                consecutive_errors = 0;
            }
            retry_after_ms = (esp_timer_get_time() / 1000LL) +
                             (consecutive_errors == 0 ? 1500 : 500);
        } else {
            consecutive_errors = 0;
            retry_after_ms = 0;
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
        if (result.blood_pressure.available) {
            detail += " bp=" + result.blood_pressure.status;
            if (result.blood_pressure.ready) {
                detail += "=" + std::to_string((int)result.blood_pressure.sbp_mmHg) + "/" +
                          std::to_string((int)result.blood_pressure.dbp_mmHg);
            } else if (!result.blood_pressure.reason.empty()) {
                detail += "(" + result.blood_pressure.reason + ")";
            }
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
    if (preview_running_.load()) {
        StopPreview();
    }
    // Join any previous threads that may have auto-stopped
    if (preview_thread_.joinable()) preview_thread_.join();
    if (detect_thread_.joinable()) detect_thread_.join();

    // 设置任务：task 为空时默认进入自动调度（每帧情绪 + 5s 坐姿 + 10s 心率），
    // 避免回退到 online 检测器持久化的单一任务导致情绪/心率任务不跑
    if (!task.empty()) {
        continuous_task_ = task;
        if (task != kTaskAutoSchedule) {
            ApplyTaskToOnlineDetector(active_detector_, task);
        }
    } else {
        continuous_task_ = kTaskAutoSchedule;
    }

    // BP quality gating needs roughly 10 FPS. Auto mode must not keep the old
    // 500ms preview cadence, otherwise the server rejects the 7s waveform.
    if (continuous_task_ == kTaskAutoSchedule && period_ms > 100) {
        period_ms = 100;
    }

    // Clear shared frame buffer
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        shared_frame_.clear();
    }

    continuous_period_ms_ = period_ms;
    continuous_running_.store(true);
    // Camera capture can run with Flash cache disabled. Keep both stacks in
    // internal memory; PSRAM-backed task stacks are not valid in that path.
    // Preview performs camera and LVGL work, while detection does not need the
    // same stack headroom. Configure them separately to reduce SRAM pressure.
    esp_pthread_cfg_t preview_cfg = esp_pthread_get_default_config();
    preview_cfg.stack_size = 8192;
    preview_cfg.prio = 2;
    esp_pthread_set_cfg(&preview_cfg);
    try {
        preview_thread_ = std::thread([this]() { this->PreviewLoop(); });
    } catch (...) {
        esp_pthread_set_cfg(nullptr);
        continuous_running_.store(false);
        ESP_LOGE(TAG, "Failed to spawn preview thread");
        return false;
    }

    esp_pthread_cfg_t detect_cfg = esp_pthread_get_default_config();
    detect_cfg.stack_size = 6144;
    detect_cfg.prio = 2;
    esp_pthread_set_cfg(&detect_cfg);
    try {
        detect_thread_ = std::thread([this]() { this->DetectionLoop(); });
    } catch (...) {
        esp_pthread_set_cfg(nullptr);
        continuous_running_.store(false);
        if (preview_thread_.joinable()) preview_thread_.join();
        ESP_LOGE(TAG, "Failed to spawn detection thread (internal SRAM may be insufficient; "
                 "free=%u B, largest=%u B)",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
        return false;
    }
    esp_pthread_set_cfg(nullptr);
    return true;
}

void VisionPipeline::StopContinuous() {
    continuous_running_.store(false);
    if (preview_thread_.joinable()) preview_thread_.join();
    if (detect_thread_.joinable()) detect_thread_.join();
}

bool VisionPipeline::IsContinuousRunning() const {
    return continuous_running_.load();
}

const std::string& VisionPipeline::GetContinuousTask() const {
    return continuous_task_;
}

// --- Preview-only (no detection) ---

void VisionPipeline::PreviewOnlyLoop() {
    ESP_LOGI(TAG, "Preview loop started (period=%ums)", preview_period_ms_);
    while (preview_running_.load()) {
        auto t0 = esp_timer_get_time();

        ImageFrame frame;
        if (!CaptureFrame(frame)) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        DetectionResult empty_result;  // no detections
        empty_result.source_width = frame.width;
        empty_result.source_height = frame.height;

        auto* lcd = ResolveLvglDisplay();
        if (lcd != nullptr && display_composer_) {
            // OV2640 直出 JPEG：解码成 RGB565 再送 LCD
            const uint8_t* disp_data = frame.data;
            int disp_w = frame.width;
            int disp_h = frame.height;
            if (frame.pixel_format == PIXFORMAT_JPEG) {
                uint8_t* dec = nullptr; size_t dec_len = 0; int dw = 0, dh = 0;
                if (DecodeJpegFrame(frame, &dec, &dec_len, &dw, &dh)) {
                    disp_data = dec; disp_w = dw; disp_h = dh;
                }
            }
            auto image = display_composer_->ComposePreview(
                disp_data, disp_w, disp_h,
                (size_t)disp_w * 2, empty_result);
            if (image) {
                lcd->SetVisionPreviewImage(std::move(image), empty_result);
            }
        }

        ReleaseCurrentFrame();

        auto t1 = esp_timer_get_time();
        int64_t elapsed_ms = (t1 - t0) / 1000LL;
        int64_t wait_ms = (int64_t)preview_period_ms_ - elapsed_ms;
        if (wait_ms > 0 && preview_running_.load()) {
            vTaskDelay(pdMS_TO_TICKS((uint32_t)wait_ms));
        }
    }
    ESP_LOGI(TAG, "Preview loop stopped");
}

bool VisionPipeline::StartPreview(uint32_t period_ms) {
    if (!initialized_) return false;
    if (continuous_running_.load()) {
        ESP_LOGW(TAG, "Cannot start preview while continuous detection is running");
        return false;
    }
    if (preview_running_.load()) {
        ESP_LOGW(TAG, "Preview already running");
        return true;
    }
    preview_period_ms_ = period_ms;
    preview_running_.store(true);
    try {
        preview_thread_ = std::thread([this]() { this->PreviewOnlyLoop(); });
    } catch (...) {
        preview_running_.store(false);
        ESP_LOGE(TAG, "Failed to spawn preview thread");
        return false;
    }
    return true;
}

void VisionPipeline::StopPreview() {
    if (!preview_running_.load()) return;
    preview_running_.store(false);
    if (preview_thread_.joinable()) {
        preview_thread_.join();
    }
}

bool VisionPipeline::IsPreviewRunning() const {
    return preview_running_.load();
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
    if (result.emotion.available && !result.emotion.label.empty()) {
        latest_emotion_ = result.emotion;
    } else if (ResultIncludesTask(result.task, kTaskFaceEmotion)) {
        latest_emotion_ = EmotionResult();
    }
    if (result.heart_rate.available) {
        latest_heart_rate_ = result.heart_rate;
        heart_rate_timestamp_ms_ = result.timestamp_ms;
    }
    if (result.blood_pressure.available) {
        latest_blood_pressure_ = result.blood_pressure;
        blood_pressure_timestamp_ms_ = result.timestamp_ms;
    }
    if (result.posture.available) {
        // "calibrated" 是标定成功的瞬时状态，不是坐姿状态，不覆盖 LCD 叠加
        if (result.posture.state != "calibrated") {
            latest_posture_ = result.posture;
            posture_timestamp_ms_ = result.timestamp_ms;
        }
    }
    if (ResultIncludesTask(result.task, kTaskFaceEmotion) &&
        result.emotion.available && !result.emotion.label.empty()) {
        // 10 帧平滑窗口：同一情绪 >=7 次判为特殊情绪，否则 neutral
        emotion_window_.push_back(result.emotion.label);
        if (emotion_window_.size() > 10) {
            emotion_window_.pop_front();
        }
        std::string majority;
        int majority_count = 0;
        for (const auto& a : emotion_window_) {
            int c = 0;
            for (const auto& b : emotion_window_) {
                if (a == b) c++;
            }
            if (c > majority_count) {
                majority_count = c;
                majority = a;
            }
        }
        if (emotion_window_.size() >= 10 && majority_count >= 7 && majority != "neutral") {
            user_emotion_ = majority;
            user_emotion_hits_ = majority_count;
        } else {
            user_emotion_ = "neutral";
            user_emotion_hits_ = 0;
        }
        user_emotion_timestamp_ms_ = result.timestamp_ms;
    } else if (ResultIncludesTask(result.task, kTaskFaceEmotion)) {
        // 服务器未检测到人脸时，立即使上一帧情绪失效，不能继续沿用旧结果。
        emotion_window_.clear();
        user_emotion_ = "neutral";
        user_emotion_hits_ = 0;
        user_emotion_timestamp_ms_ = result.timestamp_ms;
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

BloodPressureResult VisionPipeline::GetLatestBloodPressure() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return latest_blood_pressure_;
}

uint64_t VisionPipeline::GetBloodPressureTimestampMs() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return blood_pressure_timestamp_ms_;
}

PostureResult VisionPipeline::GetLatestPosture() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return latest_posture_;
}

uint64_t VisionPipeline::GetPostureTimestampMs() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return posture_timestamp_ms_;
}

std::string VisionPipeline::GetUserEmotion() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return user_emotion_;
}

int VisionPipeline::GetUserEmotionHits() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return user_emotion_hits_;
}

uint64_t VisionPipeline::GetUserEmotionTimestampMs() const {
    std::lock_guard<std::mutex> lock(sensing_mutex_);
    return user_emotion_timestamp_ms_;
}

// 播放 assets 分区中的 OGG 语音。PlaySound 同步消费数据，因此这里不保存悬空指针。
static std::mutex g_voice_alert_mutex;
static std::thread g_voice_alert_thread;

// 在独立线程中播放固定语音，避免阻塞视觉检测线程。
static void PlayVoiceAssetAsync(std::string filename, DeviceState previous_state) {
    auto& app = Application::GetInstance();
    auto& audio = app.GetAudioService();

    void* data = nullptr;
    size_t size = 0;
    if (!Assets::GetInstance().GetAssetData(filename, data, size) || data == nullptr || size == 0) {
        ESP_LOGW(TAG, "Voice asset not found: %s", filename);
        audio.EndLocalSound();
        return;
    }
    // 当前转换后的 OGG 振幅比小智 TTS 偏小，提示音使用约 +3.5 dB 增益。
    // 不关闭网络音频通道、不切换设备状态，避免与音频/SPI任务并发重配置。
    audio.PlaySound(std::string_view(static_cast<const char*>(data), size), 1.5f);

    // 等待最后一个 PCM 块真正输出后再恢复唤醒检测。
    while (!audio.IsIdle()) {
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    audio.EndLocalSound();
    const bool was_listening = previous_state == kDeviceStateListening;
    const bool was_idle = previous_state == kDeviceStateIdle;
    // 播放期间如果服务器已让设备进入说话状态，不覆盖该状态。
    if (app.GetDeviceState() == kDeviceStateSpeaking) {
        return;
    }
    if (was_listening && app.GetDeviceState() == kDeviceStateListening) {
        audio.EnableVoiceProcessing(true);
        audio.EnableWakeWordDetection(false);
    } else if (was_idle && app.GetDeviceState() == kDeviceStateIdle) {
        audio.EnableWakeWordDetection(true);
    }
}

static bool QueueVoiceAsset(const char* filename) {
    auto& app = Application::GetInstance();
    auto& audio = app.GetAudioService();
    const DeviceState previous_state = app.GetDeviceState();
    const bool was_listening = previous_state == kDeviceStateListening;

    std::lock_guard<std::mutex> lock(g_voice_alert_mutex);
    if (previous_state == kDeviceStateSpeaking || audio.IsLocalSoundPlaying() ||
        (!was_listening && !audio.IsIdle())) {
        return false;
    }
    // 回收上一条已经播放完成的线程；正在播放时由 local_sound_playing_ 拦截。
    if (g_voice_alert_thread.joinable()) {
        g_voice_alert_thread.join();
    }

    // 先占用本地语音播放权，防止多个检测结果同时创建播放线程。
    audio.BeginLocalSound();
    if (was_listening) {
        // 视觉检测继续运行，但麦克风不再把固定语音上传给服务器。
        audio.EnableVoiceProcessing(false);
    }
    audio.EnableWakeWordDetection(false);

    g_voice_alert_thread = std::thread([filename = std::string(filename), previous_state]() {
        PlayVoiceAssetAsync(filename, previous_state);
    });
    return true;
}

static void StopQueuedVoiceAlert() {
    std::lock_guard<std::mutex> lock(g_voice_alert_mutex);
    if (g_voice_alert_thread.joinable()) {
        g_voice_alert_thread.join();
    }
}

// 四类本地语音提醒：心率/血压/坐姿/情绪各自独立冷却 60 秒。
void VisionPipeline::MaybePlayVoiceAlerts(const DetectionResult& result) {
    constexpr uint64_t kAlertCooldownMs = 60000;
    constexpr float kHighHeartRateBpm = 120.0f;
    constexpr float kHighSystolicMmHg = 140.0f;
    constexpr float kHighDiastolicMmHg = 90.0f;

    const uint64_t now_ms = esp_timer_get_time() / 1000LL;

    // 健康报警优先于坐姿和情绪；每次检测最多播放一种语音。
    if (result.heart_rate.available && result.heart_rate.bpm > kHighHeartRateBpm &&
        (last_heart_alert_ms_ == 0 || now_ms - last_heart_alert_ms_ >= kAlertCooldownMs)) {
        if (QueueVoiceAsset("heart_alarm.ogg")) last_heart_alert_ms_ = now_ms;
        return;
    }

    if (result.blood_pressure.ready &&
        (result.blood_pressure.sbp_mmHg >= kHighSystolicMmHg ||
         result.blood_pressure.dbp_mmHg >= kHighDiastolicMmHg) &&
        (last_blood_alert_ms_ == 0 || now_ms - last_blood_alert_ms_ >= kAlertCooldownMs)) {
        if (QueueVoiceAsset("blood_alarm.ogg")) last_blood_alert_ms_ = now_ms;
        return;
    }

    if (result.posture.available && result.posture.state == "bad_posture" &&
        (last_posture_alert_ms_ == 0 || now_ms - last_posture_alert_ms_ >= kAlertCooldownMs)) {
        if (QueueVoiceAsset("pose_alarm.ogg")) last_posture_alert_ms_ = now_ms;
        return;
    }

    const std::string stable_emotion = GetUserEmotion();
    if (ResultIncludesTask(result.task, kTaskFaceEmotion) && stable_emotion != "" &&
        stable_emotion != "neutral" && stable_emotion != "contempt" &&
        (last_emotion_alert_ms_ == 0 || now_ms - last_emotion_alert_ms_ >= kAlertCooldownMs)) {
        const std::string filename = stable_emotion + ".ogg";
        if (QueueVoiceAsset(filename.c_str())) {
            last_emotion_alert_ms_ = now_ms;
        }
    }
}


// ---------- MCP Tools Registration ----------

static const char* kDetectorTypeOnline = "online";
#ifndef VISION_DISABLE_LOCAL_FACE
static const char* kDetectorTypeFace = "face";
#endif

static DetectorType DetectorTypeFromString(const std::string& s) {
#ifndef VISION_DISABLE_LOCAL_FACE
    if (s == kDetectorTypeFace) return DetectorType::kFace;
#endif
    if (s == kDetectorTypeOnline) return DetectorType::kOnline;
    return DetectorType::kNone;
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
    if (r.blood_pressure.available) {
        cJSON* bp = cJSON_AddObjectToObject(j, "blood_pressure");
        cJSON_AddStringToObject(bp, "status", r.blood_pressure.status.c_str());
        cJSON_AddStringToObject(bp, "reason", r.blood_pressure.reason.c_str());
        cJSON_AddNumberToObject(bp, "duration_s", r.blood_pressure.duration_s);
        cJSON_AddNumberToObject(bp, "required_window_s", r.blood_pressure.required_window_s);
        cJSON_AddBoolToObject(bp, "experimental_only", true);
        if (r.blood_pressure.ready) {
            cJSON_AddNumberToObject(bp, "sbp_mmHg", r.blood_pressure.sbp_mmHg);
            cJSON_AddNumberToObject(bp, "dbp_mmHg", r.blood_pressure.dbp_mmHg);
        }
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
        "Must be called before other self.vision.* tools. Safe to call multiple times.\n"
        "NOTE: all self.vision.* tools process images LOCALLY on the device unless the online detector "
        "is explicitly configured and selected.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return VisionPipeline::GetInstance().Initialize();
        });

    /* [tool-limit] hidden from LLM tool list; implementation retained
    mcp.AddTool("self.vision.shutdown",
        "Shutdown the vision pipeline, release detectors, stop continuous loop if running.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().Deinitialize();
            return true;
        }); */

    mcp.AddTool("self.vision.set_active_detector",
        "Choose which detector is used for detection.\n"
        "Note: `online` is the default after initialization.\n"
        "Args:\n"
        "  `detector`: One of `face` (local, low-latency, ESP-SR based, NO upload) or "
        "`online` (HTTP server based, uploads the frame to the configured server).",
        PropertyList({
            Property("detector", kPropertyTypeString)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto type = DetectorTypeFromString(p["detector"].value<std::string>());
            if (type == DetectorType::kNone) {
                throw std::runtime_error("Invalid detector name, expected `face` or `online`");
            }
            return VisionPipeline::GetInstance().SetActiveDetector(type);
        });

    /* [tool-limit] hidden from LLM tool list; implementation retained
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
        }); */

    mcp.AddTool("self.vision.detect_once",
        "Capture ONE frame from the camera, run the active detector LOCALLY, and draw detection boxes "
        "on the LCD preview image. The frame never leaves the device (no upload).\n"
        "Use for a single snapshot check. For continuous REAL-TIME detection, call `self.vision.start_continuous` instead "
        "(do not call this tool repeatedly in a loop).\n"
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

    mcp.AddTool("self.vision.get_posture",
        "Get the latest posture result, cached from the most recent `posture` task "
        "(updated every ~5s in auto mode). Does NOT trigger a new detection.\n"
        "Returns state (normal/bad_posture/unknown), reason and age_ms.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            cJSON* j = cJSON_CreateObject();
            PostureResult pose = pipe.GetLatestPosture();
            if (pose.available) {
                cJSON_AddStringToObject(j, "state", pose.state.c_str());
                cJSON_AddStringToObject(j, "reason", pose.reason.c_str());
            } else {
                cJSON_AddStringToObject(j, "state", "");
                cJSON_AddStringToObject(j, "reason", "no posture measurement yet");
            }
            uint64_t ts = pipe.GetPostureTimestampMs();
            uint64_t age_ms = (ts == 0) ? 0 : (esp_timer_get_time() / 1000LL - ts);
            cJSON_AddNumberToObject(j, "age_ms", (double)age_ms);
            return j;
        });

    mcp.AddTool("self.vision.get_blood_pressure",
        "Get the latest experimental camera blood-pressure status without capturing a frame. "
        "A number is available only after a stable high-rate rPPG window and a configured "
        "server-side BP model; never use it for medical decisions.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            BloodPressureResult bp = pipe.GetLatestBloodPressure();
            cJSON* j = cJSON_CreateObject();
            cJSON_AddBoolToObject(j, "experimental_only", true);
            if (bp.available) {
                cJSON_AddStringToObject(j, "status", bp.status.c_str());
                cJSON_AddStringToObject(j, "reason", bp.reason.c_str());
                cJSON_AddNumberToObject(j, "duration_s", bp.duration_s);
                cJSON_AddNumberToObject(j, "required_window_s", bp.required_window_s);
                if (bp.ready) {
                    cJSON_AddNumberToObject(j, "sbp_mmHg", bp.sbp_mmHg);
                    cJSON_AddNumberToObject(j, "dbp_mmHg", bp.dbp_mmHg);
                }
                uint64_t ts = pipe.GetBloodPressureTimestampMs();
                uint64_t age_ms = (ts == 0) ? 0 : (esp_timer_get_time() / 1000LL - ts);
                cJSON_AddNumberToObject(j, "age_ms", (double)age_ms);
            } else {
                cJSON_AddStringToObject(j, "status", "not_started");
            }
            return j;
        });

    mcp.AddTool("self.vision.start_blood_pressure",
        "Start automatic experimental camera blood-pressure detection. This starts the online "
        "continuous detector in blood_pressure mode and collects the server's configured rPPG window. "
        "It replaces any currently running continuous vision task.",
        PropertyList({
            Property("period_ms", kPropertyTypeInteger, 100, 50, 1000)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();
            if (pipe.IsContinuousRunning()) pipe.StopContinuous();
            uint32_t period = (uint32_t)p["period_ms"].value<int>();
            bool ok = pipe.StartContinuous(period, kTaskBloodPressure);
            cJSON* j = cJSON_CreateObject();
            cJSON_AddBoolToObject(j, "enabled", ok);
            cJSON_AddStringToObject(j, "task", kTaskBloodPressure);
            cJSON_AddNumberToObject(j, "period_ms", period);
            cJSON_AddBoolToObject(j, "experimental_only", true);
            return j;
        });

    mcp.AddTool("self.vision.stop_blood_pressure",
        "Stop experimental blood-pressure detection. If auto continuous mode is running, this "
        "stops that continuous vision session as well.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            bool was_running = pipe.IsContinuousRunning();
            pipe.StopContinuous();
            return was_running;
        });

    mcp.AddTool("self.vision.get_blood_pressure_status",
        "Return whether blood-pressure detection is enabled/running and the latest experimental result. "
        "Does not capture a frame or start a measurement.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            BloodPressureResult bp = pipe.GetLatestBloodPressure();
            bool running = pipe.IsContinuousRunning();
            const std::string& task = pipe.GetContinuousTask();
            bool enabled = running && (task == kTaskAutoSchedule || task == kTaskBloodPressure);
            cJSON* j = cJSON_CreateObject();
            cJSON_AddBoolToObject(j, "enabled", enabled);
            cJSON_AddBoolToObject(j, "running", running);
            cJSON_AddStringToObject(j, "continuous_task", task.c_str());
            cJSON_AddBoolToObject(j, "experimental_only", true);
            cJSON_AddStringToObject(j, "status", bp.available ? bp.status.c_str() : "not_started");
            if (bp.available) {
                cJSON_AddStringToObject(j, "reason", bp.reason.c_str());
                cJSON_AddNumberToObject(j, "duration_s", bp.duration_s);
                cJSON_AddNumberToObject(j, "required_window_s", bp.required_window_s);
                if (bp.ready) {
                    cJSON_AddNumberToObject(j, "sbp_mmHg", bp.sbp_mmHg);
                    cJSON_AddNumberToObject(j, "dbp_mmHg", bp.dbp_mmHg);
                }
            }
            return j;
        });

    mcp.AddTool("self.vision.get_user_emotion",
        "Get the user's current emotion, smoothed over the last 10 detection frames: "
        "a single emotion needs >=7 hits in the window to be reported, otherwise it "
        "returns `neutral`. Does NOT trigger a new detection.\n"
        "Returns emotion (one of angry/contempt/disgust/fear/happy/neutral/sad/surprise), "
        "hits, and age_ms (how old the estimate is).\n"
        "Call this to understand the user's mood before/while answering.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            cJSON* j = cJSON_CreateObject();
            cJSON_AddStringToObject(j, "emotion", pipe.GetUserEmotion().c_str());
            cJSON_AddNumberToObject(j, "hits", pipe.GetUserEmotionHits());
            uint64_t ts = pipe.GetUserEmotionTimestampMs();
            uint64_t age_ms = (ts == 0) ? 0 : (esp_timer_get_time() / 1000LL - ts);
            cJSON_AddNumberToObject(j, "age_ms", (double)age_ms);
            return j;
        });

    mcp.AddTool("self.vision.start_continuous",
        "Start REAL-TIME detection loop: capture -> detect LOCALLY -> draw boxes on LCD, repeated continuously.\n"
        "Use this when the user asks for real-time / continuous / live face detection, or wants the device to "
        "keep watching the camera (e.g. \"实时人脸检测\", \"持续盯着我\", \"检测到人脸就告诉我\").\n"
        "LOCAL-ONLY: every frame is processed on the device, nothing is uploaded to any server.\n"
        "While this loop is running, `self.camera.take_photo` (which uploads to the cloud) is blocked.\n"
        "Args:\n"
        "  `period_ms` (optional, default 500): target time between iterations in ms. "
        "Actual rate may be lower if detector is slow.\n"
        "  `task` (optional, default `auto`): for the online detector, one of "
        "`face_emotion`, `posture`, `heart_rate`, `blood_pressure`, or `auto`.\n"
        "  `auto` (recommended) schedules: face_emotion every frame, posture every 5s, "
        "heart_rate every 10s, and blood_pressure about every 2s while face frames continuously "
        "feed the rPPG buffer. If `task` is omitted, `auto` is used.",
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
        "Stop the REAL-TIME detection loop started by `self.vision.start_continuous`. "
        "Call this when the user no longer wants continuous face detection, or before switching to a single snapshot.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().StopContinuous();
            return true;
        });

    /* [tool-limit] hidden from LLM tool list; implementation retained
    mcp.AddTool("self.vision.is_continuous_running",
        "Check if the continuous loop is currently running.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return (bool)VisionPipeline::GetInstance().IsContinuousRunning();
        }); */

    mcp.AddTool("self.vision.start_preview",
        "Start live camera preview on LCD (capture + display, NO face detection). "
        "Use this to verify the camera → LCD pipeline without any AI overhead. "
        "Much lighter than start_continuous — does not run any model.\n"
        "Args:\n"
        "  `period_ms` (optional, default 200): target interval between frames.",
        PropertyList({
             Property("period_ms", kPropertyTypeInteger, 200, 50, 5000)
         }),
        [](const PropertyList& properties) -> ReturnValue {
             uint32_t period = 200;
             try {
                 period = (uint32_t)properties["period_ms"].value<int>();
             } catch (...) {
                 // use default
             }
             if (!VisionPipeline::GetInstance().StartPreview(period)) {
                throw std::runtime_error("Failed to start preview (pipeline not initialized?)");
            }
            return std::string("Preview started successfully. Call self.vision.stop_preview to stop.");
        });

    mcp.AddTool("self.vision.stop_preview",
        "Stop the live camera preview started by `self.vision.start_preview`.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            VisionPipeline::GetInstance().StopPreview();
            return true;
        });

    /* [tool-limit] hidden from LLM tool list; implementation retained
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
        }); */

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

    // Remote detector config is retained in code for compatibility, but the
    // current unified server uses online_detector for all detection tasks.
    /* [tool-limit] disabled: use self.vision.online_detector.configure instead
    mcp.AddTool("self.vision.remote_detector.configure",
        "Configure the remote HTTP detection server.\n"
        "The same server URL is also used automatically for Mijia lamp control; no separate lamp configuration is needed.\n"
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
            SyncLampServerUrl(p["url"].value<std::string>());
            return true;
        }); */

    /* [tool-limit] hidden from LLM tool list; implementation retained
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
        }); */

    // Online detector config
    mcp.AddTool("self.vision.online_detector.configure",
        "Configure the online detection server (JSON+Base64 protocol) on the LAN.\n"
        "The same server URL is also used automatically for Mijia lamp control; no separate lamp configuration is needed.\n"
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
            SyncLampServerUrl(url);
            return true;
        });

    /* [tool-limit] hidden from LLM tool list; implementation retained
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
        }); */

    mcp.AddTool("self.vision.online_detector.set_task",
        "Set the task for the online detection server (saved to flash, restored after reboot).\n"
        "Args:\n"
        "  `task`: one of `face_emotion` (face + emotion, every frame), `posture` (5s), "
        "`heart_rate` (10s), or `blood_pressure` (experimental; sustained high-rate frames required).",
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

    mcp.AddTool("self.vision.posture.calibrate",
        "Calibrate the posture baseline: record the CURRENT posture as the reference "
        "('sitting straight') baseline on the server, then run one posture detection "
        "immediately. Ask the user to sit straight first.\n"
        "The server response state is `calibrated` when successful.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            pipe.Initialize();  // auto-init if not done yet
            auto* od = pipe.GetOnlineDetector();
            if (od == nullptr) {
                cJSON* j = cJSON_CreateObject();
                cJSON_AddBoolToObject(j, "success", false);
                cJSON_AddStringToObject(j, "message", "online detector unavailable");
                return j;
            }
            od->SetCalibrateOnce(true);
            // 连续检测运行时摄像头忙，不取帧：标志会在下一个 posture 周期（≤5s）自动生效
            if (pipe.IsContinuousRunning()) {
                cJSON* j = cJSON_CreateObject();
                cJSON_AddBoolToObject(j, "success", true);
                cJSON_AddStringToObject(j, "message",
                    "calibration armed - will apply on the next posture cycle (within ~5s)");
                return j;
            }
            std::string prev_task = od->GetTask();
            std::string info;
            bool ok = pipe.OneShotDetect(true, &info, kTaskPosture);
            od->SetTask(prev_task);  // 恢复原任务，避免校准把持久化任务改成 posture
            cJSON* j = DetectionsToJson(pipe.GetLastResult());
            cJSON_AddBoolToObject(j, "success", ok);
            cJSON_AddStringToObject(j, "debug", info.c_str());
            return j;
        });

    ESP_LOGI(TAG, "Vision MCP tools registered");
}
