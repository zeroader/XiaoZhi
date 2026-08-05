#ifndef VISION_PIPELINE_H
#define VISION_PIPELINE_H

#include "detector.h"
#include "face_detector.h"
#include "remote_detector.h"
#include "vision_display.h"

#include <string>
#include <memory>
#include <atomic>
#include <thread>

#include <esp_camera.h>

class Camera;
class LvglDisplay;
class Esp32Camera;

enum class DetectorType {
    kNone = 0,
    kFace = 1,
    kRemote = 2,
};

struct PipelineStats {
    uint64_t capture_count;
    uint64_t detect_count;
    uint64_t display_count;
    uint64_t total_capture_ms;
    uint64_t total_detect_ms;
    uint64_t total_compose_ms;
    uint64_t total_display_ms;
    uint32_t last_detection_count;
    uint32_t loop_period_ms;

    PipelineStats();
    void Reset();
};

class VisionPipeline {
public:
    VisionPipeline();
    ~VisionPipeline();

    static VisionPipeline& GetInstance();

    bool Initialize();

    void Deinitialize();

    Detector* GetDetector(DetectorType type);

    DetectorType GetActiveDetectorType() const;
    bool SetActiveDetector(DetectorType type);

    FaceDetector* GetFaceDetector();
    RemoteDetector* GetRemoteDetector();
    VisionDisplay* GetDisplayComposer();

    bool OneShotDetect(bool show_on_lcd = true, std::string* out_debug_info = nullptr);

    bool StartContinuous(uint32_t period_ms = 500);
    void StopContinuous();
    bool IsContinuousRunning() const;

    // Preview-only: capture + display, NO detection (for debugging camera pipeline)
    bool StartPreview(uint32_t period_ms = 200);
    void StopPreview();
    bool IsPreviewRunning() const;

    const PipelineStats& GetStats() const;
    void ResetStats();

    bool SetCameraMirror(bool h_mirror, bool v_flip);

    const DetectionResult& GetLastResult() const;

private:
    bool initialized_;
    bool has_camera_;

    std::unique_ptr<FaceDetector> face_detector_;
    std::unique_ptr<RemoteDetector> remote_detector_;
    std::unique_ptr<VisionDisplay> display_composer_;

    DetectorType active_detector_type_;
    Detector* active_detector_;

    DetectionResult last_result_;
    PipelineStats stats_;

    std::atomic<bool> continuous_running_;
    std::thread continuous_thread_;
    uint32_t continuous_period_ms_;

    std::atomic<bool> preview_running_;
    std::thread preview_thread_;
    uint32_t preview_period_ms_;

    Esp32Camera* ResolveEsp32Camera();
    LvglDisplay* ResolveLvglDisplay();
    void ContinuousLoop();
    void PreviewLoop();   // capture + display only, no detection

    bool CaptureFrame(ImageFrame& out_frame);
    bool ReleaseCurrentFrame();
    camera_fb_t* GetCameraFb();
};

void RegisterVisionMcpTools();

#endif
