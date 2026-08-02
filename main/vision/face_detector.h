#ifndef VISION_FACE_DETECTOR_H
#define VISION_FACE_DETECTOR_H

#include "detector.h"

class FaceDetector : public Detector {
public:
    FaceDetector();
    ~FaceDetector() override;

    const char* GetName() const override;
    bool Initialize() override;
    void Deinitialize() override;
    DetectionResult Detect(const ImageFrame& frame) override;

    void SetConfidenceThreshold(float threshold);
    float GetConfidenceThreshold() const;

private:
    bool initialized_;
    float confidence_threshold_;
    void* sr_handle_;
};

#endif
