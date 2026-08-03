#ifndef VISION_FACE_DETECTOR_H
#define VISION_FACE_DETECTOR_H

#include "detector.h"

// Forward declaration from esp-dl (human_face_detect.hpp)
class HumanFaceDetect;

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
    // Convert camera RGB565 frame to RGB888 for model input
    static void Rgb565ToRgb888(const uint8_t* src, uint8_t* dst, int width, int height);

    // Nearest-neighbor resize for RGB888 (camera frame → model input 160×120)
    static void ResizeRgb888(const uint8_t* src, int src_w, int src_h,
                             uint8_t* dst, int dst_w, int dst_h);

    // Model input size for MSR_S8_V1 (two-stage face detector)
    // HumanFaceDetect::run() handles internal resize, no manual scaling needed.
    static constexpr int kModelWidth = 160;
    static constexpr int kModelHeight = 120;

    bool initialized_;
    float confidence_threshold_;
    HumanFaceDetect* detector_;  // esp-dl face detector instance
};

#endif
