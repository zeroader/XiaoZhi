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
    , continuous_period_ms_(500) {
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

    face_detector_ = std::make_unique<FaceDetector>();
    if (!face_detector_->Initialize()) {
        ESP_LOGW(TAG, "Face detector initialize returned false");
    }

    remote_detector_ = std::make_unique<RemoteDetector>();
    if (!remote_detector_->Initialize()) {
        ESP_LOGW(TAG, "Remote detector initialize returned false");
    }

    display_composer_ = std::make_unique<VisionDisplay>();

    active_detector_type_ = DetectorType::kFace;
    active_detector_ = face_detector_.get();

    initialized_ = true;
    ESP_LOGI(TAG, "Vision pipeline initialized (active=%s, camera=%s)",
             active_detector_->GetName(),
             has_camera_ ? "ok" : "none");
    return true;
}

void VisionPipeline::Deinitialize() {
    if (!initialized_) return;

    StopContinuous();

    if (face_detector_) {
        face_detector_->Deinitialize();
    }
    if (remote_detector_) {
        remote_detector_->Deinitialize();
    }

    active_detector_ = nullptr;
    active_detector_type_ = DetectorType::kNone;
    initialized_ = false;
    ESP_LOGI(TAG, "Vision pipeline deinitialized");
}

Detector* VisionPipeline::GetDetector(DetectorType type) {
    switch (type) {
        case DetectorType::kFace:   return face_detector_.get();
        case DetectorType::kRemote: return remote_detector_.get();
        default: return nullptr;
    }
}

DetectorType VisionPipeline::GetActiveDetectorType() const { return active_detector_type_; }

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

FaceDetector* VisionPipeline::GetFaceDetector() { return face_detector_.get(); }
RemoteDetector* VisionPipeline::GetRemoteDetector() { return remote_detector_.get(); }
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

bool VisionPipeline::OneShotDetect(bool show_on_lcd, std::string* out_debug_info) {
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
             "detector=%s, detections=%d, dimensions=%dx%d, total=%lldms, detect=%lldms, shown=%s",
             active_detector_->GetName(),
             (int)result.detections.size(),
             frame.width, frame.height,
             (long long)((t_all1 - t_all) / 1000LL),
             (long long)((t_detect1 - t_detect0) / 1000LL),
             displayed ? "true" : "false");
    ESP_LOGI(TAG, "OneShotDetect: %s", buf);
    if (out_debug_info) *out_debug_info = buf;
    return true;
}

void VisionPipeline::ContinuousLoop() {
    ESP_LOGI(TAG, "Continuous detection loop started (period=%ums)", continuous_period_ms_);

    uint64_t last_iter_start = 0;
    while (continuous_running_.load()) {
        auto t0 = esp_timer_get_time();
        stats_.loop_period_ms = last_iter_start == 0 ? 0
            : (uint32_t)((t0 - last_iter_start) / 1000LL);
        last_iter_start = t0;

        std::string info;
        OneShotDetect(true, &info);

        auto t1 = esp_timer_get_time();
        int64_t elapsed_ms = (t1 - t0) / 1000LL;
        int64_t wait_ms = (int64_t)continuous_period_ms_ - elapsed_ms;
        if (wait_ms > 0 && continuous_running_.load()) {
            vTaskDelay(pdMS_TO_TICKS((uint32_t)wait_ms));
        }
    }
    ESP_LOGI(TAG, "Continuous detection loop stopped");
}

bool VisionPipeline::StartContinuous(uint32_t period_ms) {
    if (!initialized_) return false;
    if (continuous_running_.load()) {
        ESP_LOGW(TAG, "Continuous detect already running");
        return true;
    }
    continuous_period_ms_ = period_ms;
    continuous_running_.store(true);
    try {
        continuous_thread_ = std::thread([this]() { this->ContinuousLoop(); });
    } catch (...) {
        continuous_running_.store(false);
        ESP_LOGE(TAG, "Failed to spawn continuous detection thread");
        return false;
    }
    return true;
}

void VisionPipeline::StopContinuous() {
    if (!continuous_running_.load()) return;
    continuous_running_.store(false);
    if (continuous_thread_.joinable()) {
        continuous_thread_.join();
    }
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


// ---------- MCP Tools Registration ----------

static const char* kDetectorTypeFace = "face";
static const char* kDetectorTypeRemote = "remote";

static DetectorType DetectorTypeFromString(const std::string& s) {
    if (s == kDetectorTypeFace) return DetectorType::kFace;
    if (s == kDetectorTypeRemote) return DetectorType::kRemote;
    return DetectorType::kNone;
}

static std::string DetectorTypeToString(DetectorType t) {
    switch (t) {
        case DetectorType::kFace:   return kDetectorTypeFace;
        case DetectorType::kRemote: return kDetectorTypeRemote;
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
        "Args:\n"
        "  `detector`: One of `face` (local, low-latency, ESP-SR based), `remote` (HTTP server based).",
        PropertyList({
            Property("detector", kPropertyTypeString)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto type = DetectorTypeFromString(p["detector"].value<std::string>());
            if (type == DetectorType::kNone) {
                throw std::runtime_error("Invalid detector name, expected `face` or `remote`");
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
            cJSON_AddItemToArray(avail, cJSON_CreateString(kDetectorTypeFace));
            cJSON_AddItemToArray(avail, cJSON_CreateString(kDetectorTypeRemote));
            return j;
        });

    mcp.AddTool("self.vision.detect_once",
        "Capture one frame from the camera, run the active detector, and draw detection boxes "
        "on the LCD preview image. Returns full detection result JSON.\n"
        "Args:\n"
        "  `show_on_lcd` (optional, default true): if false, do not update the LCD.\n",
        PropertyList({
            Property("show_on_lcd", kPropertyTypeBoolean, true)
        }),
        [](const PropertyList& p) -> ReturnValue {
            auto& pipe = VisionPipeline::GetInstance();
            std::string info;
            bool ok = pipe.OneShotDetect(p["show_on_lcd"].value<bool>(), &info);
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

    mcp.AddTool("self.vision.start_continuous",
        "Start continuous detection loop: capture -> detect -> show on LCD, repeated.\n"
        "Args:\n"
        "  `period_ms` (optional, default 500): target time between iterations in ms. "
        "Actual rate may be lower if detector is slow.",
        PropertyList({
            Property("period_ms", kPropertyTypeInteger, 500, 100, 60000)
        }),
        [](const PropertyList& p) -> ReturnValue {
            uint32_t period = (uint32_t)p["period_ms"].value<int>();
            return VisionPipeline::GetInstance().StartContinuous(period);
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

    // Remote detector config
    mcp.AddTool("self.vision.remote_detector.configure",
        "Configure the remote HTTP detection server.\n"
        "This detector POSTs the camera frame as a multipart/form-data request (fields: `image` = JPEG, "
        "`width` = src_w, `height` = src_h).\n"
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
            auto* rd = VisionPipeline::GetInstance().GetRemoteDetector();
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

    ESP_LOGI(TAG, "Vision MCP tools registered");
}
