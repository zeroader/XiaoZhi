#ifndef VISION_ONLINE_DETECTOR_H
#define VISION_ONLINE_DETECTOR_H

#include "detector.h"
#include <cstddef>
#include <string>
#include <vector>

class OnlineDetector : public Detector {
public:
    OnlineDetector();
    ~OnlineDetector() override;

    const char* GetName() const override;
    bool Initialize() override;
    void Deinitialize() override;
    DetectionResult Detect(const ImageFrame& frame) override;
    DetectionResult Detect(const ImageFrame& frame, const std::vector<std::string>& tasks);

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
    static size_t JpegOutputCallback(void* arg, size_t index, const void* data, size_t len);
    bool EnsureJpegBuffer(size_t required);
    size_t AppendJpegData(size_t index, const void* data, size_t len);

    bool initialized_;
    std::string endpoint_url_;
    int timeout_sec_;
    int jpeg_quality_;
    std::string task_;       // 当前任务：face_emotion / posture / heart_rate / blood_pressure
    int64_t frame_id_;       // 自增帧号
    bool auto_frame_id_;     // 是否每次请求自动递增 frame_id
    bool calibrate_once_;    // 下一次 posture 请求携带 calibrate=true（坐姿基准标定）

    // RGB565 软编码输出缓冲。检测循环是单一消费者，编码完成后直接复用，
    // 避免每帧创建编码线程、FreeRTOS 队列和大量 JPEG 临时块。
    uint8_t* jpeg_buffer_ = nullptr;
    size_t jpeg_buffer_cap_ = 0;

    // Parse JSON response: new unified format (result + performance) or legacy {detections}
    bool ParseResponse(const std::string& json_str, DetectionResult& result);
};

#endif
