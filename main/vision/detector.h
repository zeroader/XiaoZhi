#ifndef VISION_DETECTOR_H
#define VISION_DETECTOR_H

#include <string>
#include <vector>
#include <cstdint>

// 任务类型（与 server/protocol.py 保持一致）
const char kTaskFaceEmotion[] = "face_emotion";  // 人脸 + 情绪（每帧）
const char kTaskPosture[] = "posture";           // 坐姿（5s 一次）
const char kTaskHeartRate[] = "heart_rate";      // 心率（10s 一次）
const char kTaskAutoSchedule[] = "auto";         // 连续模式下按 每帧/5s/10s 自动调度

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

// 情绪结果（face_emotion 任务）
struct EmotionResult {
    std::string label;      // angry/contempt/disgust/fear/happy/neutral/sad/surprise
    float confidence;
    bool available;         // false = 无人脸或情绪模型不可用

    EmotionResult() : confidence(0.0f), available(false) {}
};

// 坐姿结果（posture 任务）
struct PostureResult {
    std::string state;      // normal / bad_posture / 空
    std::string reason;     // ok / head_forward / ...
    bool available;

    PostureResult() : available(false) {}
};

// 心率结果（heart_rate 任务）
struct HeartRateResult {
    float bpm;
    float confidence;
    float fs;
    int frames_used;
    bool available;
    std::string error_message;  // 如 "insufficient_frames"

    HeartRateResult() : bpm(0.0f), confidence(0.0f), fs(0.0f), frames_used(0), available(false) {}
};

struct BloodPressureResult {
    float systolic;
    float diastolic;
    bool available;

    BloodPressureResult() : systolic(0.0f), diastolic(0.0f), available(false) {}
};

struct DetectionResult {
    std::vector<Detection> detections;
    int source_width;
    int source_height;
    uint64_t timestamp_ms;
    bool connection_ok;       // false = server unreachable or HTTP error
    std::string error_message; // detail when connection_ok is false

    // 新协议字段
    int64_t frame_id;         // 服务器回显的帧号
    std::string task;         // 本次请求的任务类型
    EmotionResult emotion;
    PostureResult posture;
    HeartRateResult heart_rate;
    BloodPressureResult blood_pressure;
    float decode_ms;          // 服务器解码耗时
    float infer_ms;           // 服务器推理耗时
    float total_ms;           // 服务器总耗时

    DetectionResult()
        : source_width(0), source_height(0), timestamp_ms(0), connection_ok(true),
          frame_id(-1), decode_ms(0.0f), infer_ms(0.0f), total_ms(0.0f) {}
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
