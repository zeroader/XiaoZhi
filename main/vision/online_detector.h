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

    const std::string& GetEndpoint() const;
    int GetTimeoutSec() const;
    int GetJpegQuality() const;

private:
    bool initialized_;
    std::string endpoint_url_;
    int timeout_sec_;
    int jpeg_quality_;

    // Encode binary data to base64 string
    static std::string Base64Encode(const uint8_t* data, size_t len);

    // Build JSON request body: {"type":"face","images":["base64..."],"width":320,"height":240,"format":"jpeg"}
    std::string BuildRequestBody(const std::string& base64_image, int width, int height);

    // Parse JSON response with bbox detections
    bool ParseResponse(const std::string& json_str, DetectionResult& result);
};

#endif
