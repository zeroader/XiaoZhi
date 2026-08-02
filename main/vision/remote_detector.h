#ifndef VISION_REMOTE_DETECTOR_H
#define VISION_REMOTE_DETECTOR_H

#include "detector.h"
#include <string>

class RemoteDetector : public Detector {
public:
    RemoteDetector();
    ~RemoteDetector() override;

    const char* GetName() const override;
    bool Initialize() override;
    void Deinitialize() override;
    DetectionResult Detect(const ImageFrame& frame) override;

    void SetEndpoint(const std::string& url);
    void SetAuthToken(const std::string& token);
    void SetRequestTimeoutSec(int sec);
    void SetExtraHeader(const std::string& key, const std::string& value);

    const std::string& GetEndpoint() const;
    int GetRequestTimeoutSec() const;

private:
    bool initialized_;
    std::string endpoint_url_;
    std::string auth_token_;
    int timeout_sec_;
    std::string extra_headers_;

    std::string EncodeMultipartBody(const uint8_t* jpeg_data, size_t jpeg_len,
                                    const std::string& boundary,
                                    int src_w, int src_h);
    bool ParseResponseJson(const std::string& json_str, DetectionResult& result);
};

#endif
