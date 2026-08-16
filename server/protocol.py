"""
ESP32 生物特征感知服务器 - 通信协议定义

Request (POST /detect):
  {
    "frame_id": 12345,
    "task": "face_emotion",          # face_emotion | posture | heart_rate
    "image": {
      "data": "base64 jpeg",
      "width": 320,
      "height": 240,
      "format": "jpeg"
    }
  }

兼容旧协议（人脸检测专用，ESP32 online_detector 仍在用）:
  {
    "type": "face",
    "images": ["base64_encoded_jpeg"],
    "width": 320,
    "height": 240,
    "format": "jpeg"
  }
  响应: { "detections": [ { "class": "face", "confidence": ..., "bbox": {...} } ] }

Response (新协议):
  {
    "frame_id": 12345,
    "task": "face_emotion",
    "result": { ... },
    "performance": { "decode_ms": ..., "infer_ms": ..., "total_ms": ... }
  }

所有周期任务由 ESP32 控制请求频率，服务器不进行任何定时任务。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================
# task 类型
# ============================================================

TASK_FACE_EMOTION = "face_emotion"   # 人脸 + 情绪（每帧）
TASK_POSTURE = "posture"             # 坐姿（5s 一次）
TASK_HEART_RATE = "heart_rate"       # 心率（10s 一次）

VALID_TASKS = {TASK_FACE_EMOTION, TASK_POSTURE, TASK_HEART_RATE}

# 情绪类别（AffectNet 8 分类，与 enet_b0_8_best_afew 模型对应）
EMOTION_LABELS = ["angry", "contempt", "disgust", "fear",
                  "happy", "neutral", "sad", "surprise"]


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ImageRequest:
    """单张图片请求"""
    data: str                    # base64 编码的 JPEG
    width: int = 0               # 原始宽度（用于坐标缩放）
    height: int = 0              # 原始高度
    format: str = "jpeg"


@dataclass
class DetectRequest:
    """新协议请求"""
    frame_id: int
    task: str
    image: ImageRequest
    calibrate: bool = False   # posture 任务: 是否将当前姿态记录为基准坐姿


class ProtocolError(ValueError):
    """协议解析错误"""


# ============================================================
# 请求解析
# ============================================================

def parse_detect_request(payload: Dict[str, Any]) -> DetectRequest:
    """解析新协议请求。格式错误时抛出 ProtocolError。"""
    try:
        frame_id = int(payload.get("frame_id", -1))
    except (TypeError, ValueError):
        raise ProtocolError("`frame_id` must be an integer")

    task = payload.get("task", "")
    if task not in VALID_TASKS:
        raise ProtocolError(f"`task` must be one of {sorted(VALID_TASKS)}, got {task!r}")

    img = payload.get("image")
    if not isinstance(img, dict):
        raise ProtocolError("`image` must be an object {data, width, height, format}")

    data = img.get("data", "")
    if not isinstance(data, str) or not data:
        raise ProtocolError("`image.data` must be a non-empty base64 string")

    try:
        width = int(img.get("width", 0))
        height = int(img.get("height", 0))
    except (TypeError, ValueError):
        raise ProtocolError("`image.width/height` must be integers")

    fmt = img.get("format", "jpeg") or "jpeg"

    calibrate = bool(payload.get("calibrate", False))

    return DetectRequest(
        frame_id=frame_id,
        task=task,
        image=ImageRequest(data=data, width=width, height=height, format=fmt),
        calibrate=calibrate,
    )


def is_legacy_request(payload: Dict[str, Any]) -> bool:
    """判断是否为旧协议：无 task 字段，但带有 type/images。"""
    if "task" in payload:
        return False
    return "images" in payload or payload.get("type") == "face"


def parse_legacy_request(payload: Dict[str, Any]) -> List[ImageRequest]:
    """解析旧协议：images 为 base64 列表。"""
    images = payload.get("images", [])
    if not isinstance(images, list) or not images:
        raise ProtocolError("`images` must be a non-empty list")

    try:
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
    except (TypeError, ValueError):
        raise ProtocolError("`width/height` must be integers")

    fmt = payload.get("format", "jpeg") or "jpeg"
    result = []
    for b64 in images:
        if not isinstance(b64, str) or not b64:
            continue
        result.append(ImageRequest(data=b64, width=width, height=height, format=fmt))
    if not result:
        raise ProtocolError("`images` contains no valid entries")
    return result


# ============================================================
# 响应构建
# ============================================================

def build_response(frame_id: int, task: str, result: Dict[str, Any],
                   performance: Dict[str, Any]) -> Dict[str, Any]:
    """按新协议构建统一响应。"""
    return {
        "frame_id": frame_id,
        "task": task,
        "result": result,
        "performance": performance,
    }


def build_legacy_response(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按旧协议构建人脸检测响应（保持与旧 ESP32 客户端兼容）。"""
    return {"detections": detections}
