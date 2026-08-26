#ifndef VISION_ONLINE_DETECTOR_H
#define VISION_ONLINE_DETECTOR_H

#include "detector.h"
#include <string>

class OnlineDetector : public Detector {
public:
    OnlineDetector();
    ~OnlineDetector() override;

    const char* GetName() const override;
    bool Initialize() override;
    void Deinitialize() override;
    DetectionResult Detect(const ImageFrame& frame) override;

    void SetEndpoint(const std::string& url);
    void SetTimeoutSec(int sec);
    void SetJpegQuality(int quality);
    void SetTask(const std::string& task);
    void SetAutoFrameId(bool enabled);
    void SetCalibrateOnce(bool enabled);

    const std::string& GetEndpoint() const;
    int GetTimeoutSec() const;
    int GetJpegQuality() const;
    const std::string& GetTask() const;
    bool GetAutoFrameId() const;

private:
    bool initialized_;
    std::string endpoint_url_;
    int timeout_sec_;
    int jpeg_quality_;
    std::string task_;       // 当前任务：face_emotion / posture / heart_rate / blood_pressure
    int64_t frame_id_;       // 自增帧号
    bool auto_frame_id_;     // 是否每次请求自动递增 frame_id
    bool calibrate_once_;    // 下一次 posture 请求携带 calibrate=true（坐姿基准标定）

    // Parse JSON response: new unified format (result + performance) or legacy {detections}
    bool ParseResponse(const std::string& json_str, DetectionResult& result);
};

#endif
