#ifndef VISION_PIPELINE_H
#define VISION_PIPELINE_H

#include "detector.h"
#ifndef VISION_DISABLE_LOCAL_FACE
#include "face_detector.h"
#endif
#include "remote_detector.h"
#include "online_detector.h"
#include "vision_display.h"

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>

#include <esp_camera.h>

class Camera;
class LvglDisplay;
class Esp32Camera;

enum class DetectorType {
    kNone = 0,
    kFace = 1,
    kRemote = 2,
    kOnline = 3,
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

#ifndef VISION_DISABLE_LOCAL_FACE
    FaceDetector* GetFaceDetector();
#endif
    RemoteDetector* GetRemoteDetector();
    OnlineDetector* GetOnlineDetector();
    VisionDisplay* GetDisplayComposer();

    bool OneShotDetect(bool show_on_lcd = true, std::string* out_debug_info = nullptr,
                       const std::string& task = "");

    bool StartContinuous(uint32_t period_ms = 500, const std::string& task = "");
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

    // 最近一次心率结果（跨帧保留，供查询）
    HeartRateResult GetLatestHeartRate() const;
    uint64_t GetHeartRateTimestampMs() const;

    // 最近一次坐姿结果（跨帧保留，供查询）
    PostureResult GetLatestPosture() const;
    uint64_t GetPostureTimestampMs() const;

    // 用户情绪（10帧平滑：同一情绪 >=7 次判为特殊情绪，否则 neutral）
    std::string GetUserEmotion() const;
    int GetUserEmotionHits() const;
    uint64_t GetUserEmotionTimestampMs() const;

private:
    bool initialized_;
    bool has_camera_;

#ifndef VISION_DISABLE_LOCAL_FACE
    std::unique_ptr<FaceDetector> face_detector_;
#endif
    std::unique_ptr<RemoteDetector> remote_detector_;
    std::unique_ptr<OnlineDetector> online_detector_;
    std::unique_ptr<VisionDisplay> display_composer_;

    DetectorType active_detector_type_;
    Detector* active_detector_;

    DetectionResult last_result_;
    std::mutex result_mutex_;  // protects last_result_ between preview & detect threads
    PipelineStats stats_;

    std::atomic<bool> continuous_running_;
    std::thread preview_thread_;
    std::thread detect_thread_;
    uint32_t continuous_period_ms_;
    std::string continuous_task_;   // 连续检测任务：face_emotion / posture / heart_rate / auto

    // Preview 线程与 Detection 线程之间的共享帧缓冲
    std::mutex frame_mutex_;            // 保护 shared_frame_ / shared_frame_* / shared_frame_version_
    std::vector<uint8_t> shared_frame_;
    uint32_t shared_frame_w_ = 0;
    uint32_t shared_frame_h_ = 0;
    uint32_t shared_frame_version_ = 0; // 每次新帧自增，Detection 线程据此跳过重复帧

    // 跨帧缓存的感知结果：心率值供查询，坐姿用于 LCD 持续叠加
    HeartRateResult latest_heart_rate_;
    PostureResult latest_posture_;
    uint64_t heart_rate_timestamp_ms_;
    uint64_t posture_timestamp_ms_;
    mutable std::mutex sensing_mutex_;  // 保护 latest_heart_rate_ / latest_posture_ / 情绪窗口

    // 用户情绪平滑窗口（最近10帧）
    std::deque<std::string> emotion_window_;
    std::string user_emotion_;          // 平滑结果：特殊情绪或 neutral
    int user_emotion_hits_;             // 该情绪的命中次数
    uint64_t user_emotion_timestamp_ms_;

    // 坐姿提醒状态
    bool posture_reminded_ = false;      // 当前是否已处于"已提醒"的坏坐姿状态
    uint64_t last_posture_remind_ms_ = 0;  // 上次提醒时间（防刷屏）

    // Preview-only (capture + display, NO detection)：与 start_continuous 共用 preview_thread_
    std::atomic<bool> preview_running_;
    uint32_t preview_period_ms_;

    Esp32Camera* ResolveEsp32Camera();
    LvglDisplay* ResolveLvglDisplay();
    void PreviewLoop();   // capture + display + 填充共享帧缓冲（连续检测用）
    void PreviewOnlyLoop();  // capture + display only，无检测
    void DetectionLoop();

    void CacheSensingResult(const DetectionResult& result);
    void MaybeNotifyPosture(const DetectionResult& result);

    bool CaptureFrame(ImageFrame& out_frame);
    bool ReleaseCurrentFrame();
    camera_fb_t* GetCameraFb();
};

void RegisterVisionMcpTools();

#endif
