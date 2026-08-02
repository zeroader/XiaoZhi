#ifndef VISION_DETECTOR_H
#define VISION_DETECTOR_H

#include <string>
#include <vector>
#include <cstdint>

struct BoundingBox {
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int w, int h) : x(x_), y(y_), width(w), height(h) {}
};

struct Detection {
    std::string class_name;
    int class_id;
    float confidence;
    BoundingBox box;

    Detection() : class_id(-1), confidence(0.0f) {}
    Detection(const std::string& name, int id, float conf, const BoundingBox& b)
        : class_name(name), class_id(id), confidence(conf), box(b) {}
};

struct DetectionResult {
    std::vector<Detection> detections;
    int source_width;
    int source_height;
    uint64_t timestamp_ms;

    DetectionResult() : source_width(0), source_height(0), timestamp_ms(0) {}
};

struct ImageFrame {
    const uint8_t* data;
    size_t len;
    int width;
    int height;
    int pixel_format;

    ImageFrame() : data(nullptr), len(0), width(0), height(0), pixel_format(0) {}
};

class Detector {
public:
    virtual ~Detector() = default;

    virtual const char* GetName() const = 0;

    virtual bool Initialize() = 0;

    virtual void Deinitialize() = 0;

    virtual DetectionResult Detect(const ImageFrame& frame) = 0;
};

#endif
