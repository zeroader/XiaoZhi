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
    std::string task_;       // 当前任务：face_emotion / posture / heart_rate
    int64_t frame_id_;       // 自增帧号
    bool auto_frame_id_;     // 是否每次请求自动递增 frame_id

    // Encode binary data to base64 string
    static std::string Base64Encode(const uint8_t* data, size_t len);

    // Build JSON request body (new protocol):
    // {"frame_id":N,"task":"...","image":{"data":"base64...","width":W,"height":H,"format":"jpeg"}}
    std::string BuildRequestBody(const std::string& base64_image, int width, int height);

    // Parse JSON response: new unified format (result + performance) or legacy {detections}
    bool ParseResponse(const std::string& json_str, DetectionResult& result);
};

#endif
