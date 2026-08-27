"""
ESP32 生物特征感知服务器 - Flask 入口

职责:
  1. 接收图片（HTTP POST JPEG / Base64）
  2. 根据 task 执行推理（face_emotion / posture / heart_rate / blood_pressure）
  3. 保存最近 N 帧到 frame_buffer（供 rPPG 时序分析）
  4. 返回结果

核心原则:
  * 所有周期任务由 ESP32 控制请求频率，服务器不做任何定时任务
  * 服务器只保存最近 N 帧，不采集视频

启动方式:
  1. 安装依赖: pip install -r server_requirements.txt
  2. 下载模型: python main.py --download-model
  3. 启动服务: python main.py --host 0.0.0.0 --port 8291
"""

import argparse
import base64
import concurrent.futures
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

import protocol
import state as state_mod
from detectors.face_detector import FaceDetector
from detectors.emotion_detector import EmotionDetector
from detectors.heart_detector import HeartRateDetector, METHODS as HEART_METHODS
from detectors.blood_pressure_detector import BloodPressureDetector

app = Flask(__name__)

# ============================================================
# 目录与模型配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FACE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
MODEL_EMOTION = MODELS_DIR / "enet_b0_8_best_afew.onnx"
MODEL_POSE = MODELS_DIR / "yolov8n_pose.onnx"

# 模型下载地址
MODEL_FACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_POSE_URL = "https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx"
MODEL_EMOTION_URL = "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx"

# 帧缓存大小（30 帧 ≈ 2s@15fps）
BUFFER_MAXLEN = 30
# 长时 rPPG 仅缓存前额 RGB 均值，不缓存完整视频帧。
RPPG_MAXLEN = 1800  # 60s @ 30fps
# 缓存帧降采样宽度，节省内存
CACHE_MAX_WIDTH = 320


# ============================================================
# 全局实例
# ============================================================

state: state_mod.ServerState = None
face_detector: FaceDetector = None
emotion_detector: EmotionDetector = None
heart_detector: HeartRateDetector = None
blood_pressure_detector: BloodPressureDetector = None

# 旧协议专用：上一帧/响应时间（FPS 统计）
_last_frame_time = None
_last_response_time = None

# ============================================================
# 图像解码
# ============================================================

def decode_image(image: protocol.ImageRequest) -> np.ndarray:
    """Base64 -> JPEG -> RGB numpy 数组"""
    jpeg_bytes = base64.b64decode(image.data)
    return decode_jpeg_bytes(jpeg_bytes)


def decode_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> RGB numpy 数组"""
    pil_img = Image.open(io.BytesIO(jpeg_bytes))
    return np.array(pil_img.convert("RGB"))


def cache_resized(rgb: np.ndarray) -> np.ndarray:
    """降采样用于缓存的副本（省内存，供 rPPG 使用）。"""
    h, w = rgb.shape[:2]
    if w <= CACHE_MAX_WIDTH:
        return rgb
    scale = CACHE_MAX_WIDTH / float(w)
    nw = CACHE_MAX_WIDTH
    nh = max(1, int(round(h * scale)))
    import cv2
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _capture_timestamp_seconds(value) -> float:
    """Convert an optional client monotonic capture timestamp from ms to seconds."""
    if value is None or value == "":
        return time.time()
    try:
        timestamp_ms = int(value)
    except (TypeError, ValueError):
        raise ValueError("`capture_timestamp_ms` must be an integer")
    if timestamp_ms < 0:
        raise ValueError("`capture_timestamp_ms` must be non-negative")
    return timestamp_ms / 1000.0


def _cache_and_sample(rgb: np.ndarray, timestamp: float) -> int:
    """Update the short frame cache and the compact long-window rPPG sample buffer."""
    sample_rgb, face_confidence = blood_pressure_detector.sample_frame(rgb)
    # A desktop/ESP32 scheduler can issue multiple task requests for one captured
    # frame. Keep each capture only once so heart-rate timing stays well-defined.
    if not state.push_rppg_sample(timestamp, sample_rgb, face_confidence):
        return -1
    cached = cache_resized(rgb)
    return state.push_frame(cached, cached.shape[1], cached.shape[0], timestamp)


# ============================================================
# 任务分发
# ============================================================

def _detect_face_emotion(rgb: np.ndarray, src_w: int, src_h: int) -> dict:
    """人脸 + 情绪"""
    result = {"face": None, "emotion": None}

    face = face_detector.detect_primary_face(rgb, src_w, src_h)
    if face is None:
        return result
    result["face"] = face

    if emotion_detector is not None:
        bbox = face["bbox"]
        x = max(0, bbox["x"]); y = max(0, bbox["y"])
        x2 = min(rgb.shape[1], x + bbox["width"])
        y2 = min(rgb.shape[0], y + bbox["height"])
        if x2 - x > 0 and y2 - y > 0:
            crop = rgb[y:y2, x:x2]
            result["emotion"] = emotion_detector.detect(crop)
    else:
        result["emotion"] = {"error": "emotion_model_missing"}

    return result


def _detect_posture(rgb: np.ndarray, src_w: int, src_h: int) -> dict:
    """基于当前帧人脸位置和面积的简化坐姿检测。"""
    face = face_detector.detect_primary_face(rgb, src_w, src_h)
    if face is None:
        return {"state": "unknown", "reason": "no_face", "metrics": {}}

    bbox = face["bbox"]
    frame_w = max(1, src_w)
    frame_h = max(1, src_h)
    face_center_x = bbox["x"] + bbox["width"] / 2.0
    face_area_ratio = (bbox["width"] * bbox["height"]) / (frame_w * frame_h)
    center_ratio = face_center_x / frame_w

    # 画面中间水平居中的 50%；人脸面积达到四分之一即视为过大。
    if face_area_ratio >= 0.25:
        raw_state = "bad_posture"
        reason = "face_too_large"
    elif center_ratio < 0.25:
        raw_state = "bad_posture"
        reason = "face_too_left"
    elif center_ratio > 0.75:
        raw_state = "bad_posture"
        reason = "face_too_right"
    else:
        raw_state = "normal"
        reason = "ok"

    return {
        "state": raw_state,
        "reason": reason,
        "metrics": {
            "face_center_x": round(center_ratio, 4),
            "face_area_ratio": round(face_area_ratio, 4),
            "face": face,
        },
    }


def _detect_heart_rate() -> dict:
    """rPPG 心率：使用缓存帧，不重新采集"""
    frames = state.get_recent_frames()
    return heart_detector.detect(frames)


def _detect_blood_pressure() -> dict:
    """Evaluate the latest long rPPG window without recording another video sequence."""
    samples = state.get_recent_rppg_samples(blood_pressure_detector.window_seconds)
    result = blood_pressure_detector.detect(samples)
    # Blood-pressure requests also sample the current frame. Return that face
    # so the device can keep the overlay tracking while the BP window is built.
    face = blood_pressure_detector.get_last_face()
    if face is not None:
        result["face"] = face
    return result


def _dispatch_task(task: str, rgb: np.ndarray, src_w: int, src_h: int,
                   calibrate: bool = False) -> dict:
    """按 task 分发推理（JSON 与 multipart 协议共用）"""
    if task == protocol.TASK_FACE_EMOTION:
        result = _detect_face_emotion(rgb, src_w, src_h)
        state.set_face(result["face"])
        state.set_emotion(result["emotion"])
    elif task == protocol.TASK_POSTURE:
        # calibrate 参数保留兼容性；简化规则不需要校准。
        result = _detect_posture(rgb, src_w, src_h)
        state.set_posture(result)
    elif task == protocol.TASK_HEART_RATE:
        result = _detect_heart_rate()
        state.set_heart_rate(result)
    elif task == protocol.TASK_BLOOD_PRESSURE:
        result = _detect_blood_pressure()
        state.set_blood_pressure(result)
    else:
        result = {"error": f"unsupported task: {task}"}
    return result


# ============================================================
# HTTP API
# ============================================================

def _handle_multipart_detect(t_start: float, client_net_ms: float):
    """multipart/form-data 协议：二进制 JPEG 上传（免 base64）。

    请求字段:
      frame_id / task / calibrate(可选 "true") / capture_timestamp_ms(可选) 为 form 字段
      image 为文件字段（JPEG 二进制）
    响应: 与 JSON 协议相同的统一格式 {frame_id, task, result, performance}
    """
    global _last_frame_time, _last_response_time

    # 解析 form 字段
    try:
        frame_id = int(request.form.get("frame_id", -1))
    except (TypeError, ValueError):
        return jsonify({"error": "`frame_id` must be an integer"}), 400

    task = request.form.get("task", "")
    if task not in protocol.VALID_TASKS:
        return jsonify({"error": f"`task` must be one of {sorted(protocol.VALID_TASKS)}"}), 400

    calibrate = request.form.get("calibrate", "false") in ("true", "1", "True")
    try:
        capture_timestamp = _capture_timestamp_seconds(request.form.get("capture_timestamp_ms"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 读取 JPEG 文件字段
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "missing `image` file field"}), 400
    jpeg_bytes = f.read()
    if not jpeg_bytes:
        return jsonify({"error": "empty `image` file"}), 400

    # 解码
    t0 = time.time()
    try:
        rgb = decode_jpeg_bytes(jpeg_bytes)
    except Exception as e:
        return jsonify({"error": f"decode failed - {e}"}), 400
    t_decode_ms = (time.time() - t0) * 1000

    # 若未提供原始尺寸，回退到实际图像尺寸
    try:
        src_w = int(request.form.get("width", 0)) or rgb.shape[1]
        src_h = int(request.form.get("height", 0)) or rgb.shape[0]
    except (TypeError, ValueError):
        src_w, src_h = rgb.shape[1], rgb.shape[0]

    # 保存当前帧，并提取一个紧凑的前额 RGB 采样供长期 rPPG 使用。
    cache_id = _cache_and_sample(rgb, capture_timestamp)

    t1 = time.time()
    result = _dispatch_task(task, rgb, src_w, src_h, calibrate=calibrate)
    t_infer_ms = (time.time() - t1) * 1000

    # FPS 统计
    now = time.time()
    interval_ms = 0.0
    fps = 0.0
    if _last_frame_time is not None:
        interval_ms = (now - _last_frame_time) * 1000
        fps = 1000.0 / interval_ms if interval_ms > 0 else 0.0
    _last_frame_time = now
    _last_response_time = now

    t_elapsed = (time.time() - t_start) * 1000
    print(f"[{time.strftime('%H:%M:%S')}] [mp] frame={cache_id} task={task} "
          f"interval={interval_ms:.0f}ms FPS={fps:.1f} | client+net={client_net_ms:.0f}ms "
          f"server={t_elapsed:.0f}ms (decode={t_decode_ms:.0f}ms infer={t_infer_ms:.0f}ms) "
          f"jpeg={len(jpeg_bytes)}B")

    perf = {
        "decode_ms": round(t_decode_ms, 2),
        "infer_ms": round(t_infer_ms, 2),
        "total_ms": round(t_elapsed, 2),
    }
    return jsonify(protocol.build_response(frame_id, task, result, perf))


@app.route("/detect", methods=["POST"])
def detect():
    global _last_frame_time, _last_response_time
    t_start = time.time()

    # 客户端(ESP32)+网络耗时
    client_net_ms = 0.0
    if _last_response_time is not None:
        client_net_ms = (t_start - _last_response_time) * 1000

    # ---------------- multipart 协议（二进制 JPEG，免 base64） ----------------
    if request.mimetype == "multipart/form-data":
        return _handle_multipart_detect(t_start, client_net_ms)

    # ---------------- JSON 协议 ----------------
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    # ---------------- 旧协议（人脸检测专用） ----------------
    if protocol.is_legacy_request(data):
        try:
            images = protocol.parse_legacy_request(data)
        except protocol.ProtocolError as e:
            return jsonify({"error": str(e)}), 400

        t_decode_ms = 0.0
        t_infer_ms = 0.0
        all_detections = []
        for img in images:
            t0 = time.time()
            try:
                rgb = decode_image(img)
            except Exception as e:
                return jsonify({"error": f"decode failed - {e}"}), 400
            t1 = time.time()

            # 若未提供原始尺寸，回退到实际图像尺寸
            src_w = img.width or rgb.shape[1]
            src_h = img.height or rgb.shape[0]

            dets = face_detector.detect(rgb, src_w, src_h)
            t2 = time.time()

            t_decode_ms += (t1 - t0) * 1000
            t_infer_ms += (t2 - t1) * 1000
            for d in dets:
                all_detections.append({
                    "class": "face",
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                })

            # 旧协议也更新 rPPG 缓存；旧客户端没有采集时间戳，回退服务器时间。
            _cache_and_sample(rgb, time.time())

        _last_frame_time = t_start
        _last_response_time = t_start
        return jsonify(protocol.build_legacy_response(all_detections))

    # ---------------- 新协议 ----------------
    try:
        req = protocol.parse_detect_request(data)
    except protocol.ProtocolError as e:
        return jsonify({"error": str(e)}), 400

    t0 = time.time()
    try:
        rgb = decode_image(req.image)
    except Exception as e:
        return jsonify({"error": f"decode failed - {e}"}), 400
    t_decode_ms = (time.time() - t0) * 1000

    # 若未提供原始尺寸，回退到实际图像尺寸
    src_w = req.image.width or rgb.shape[1]
    src_h = req.image.height or rgb.shape[0]

    capture_timestamp = _capture_timestamp_seconds(req.capture_timestamp_ms)
    # 保存当前帧，并提取一个紧凑的前额 RGB 采样供长期 rPPG 使用。
    frame_id = _cache_and_sample(rgb, capture_timestamp)

    t1 = time.time()
    # 按 task 分发（JSON 与 multipart 共用）
    result = _dispatch_task(req.task, rgb, src_w, src_h, calibrate=req.calibrate)
    t_infer_ms = (time.time() - t1) * 1000

    # FPS 统计（新协议响应频率）
    now = time.time()
    interval_ms = 0.0
    fps = 0.0
    if _last_frame_time is not None:
        interval_ms = (now - _last_frame_time) * 1000
        fps = 1000.0 / interval_ms if interval_ms > 0 else 0.0
    _last_frame_time = now
    _last_response_time = now

    t_elapsed = (time.time() - t_start) * 1000
    print(f"[{time.strftime('%H:%M:%S')}] frame={frame_id} task={req.task} "
          f"interval={interval_ms:.0f}ms FPS={fps:.1f} | client+net={client_net_ms:.0f}ms "
          f"server={t_elapsed:.0f}ms (decode={t_decode_ms:.0f}ms infer={t_infer_ms:.0f}ms)")

    perf = {
        "decode_ms": round(t_decode_ms, 2),
        "infer_ms": round(t_infer_ms, 2),
        "total_ms": round(t_elapsed, 2),
    }
    return jsonify(protocol.build_response(req.frame_id, req.task, result, perf))


@app.route("/state", methods=["GET"])
def get_state():
    return jsonify(state.snapshot())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "face": MODEL_FACE.name,
            "emotion": MODEL_EMOTION.name if emotion_detector else None,
            "pose": "face_geometry",
            "heart_rate": "rppg (cached frames)" if heart_detector else None,
            "blood_pressure": blood_pressure_detector.model.describe()
            if blood_pressure_detector else None,
        },
        "buffer_maxlen": BUFFER_MAXLEN,
    })


# ============================================================
# 米家台灯控制（MIoT，需 python-miio）
# ============================================================

_lamp_controller = None
_lamp_error = None
_lamp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_lamp_future = None
LAMP_REQUEST_TIMEOUT_SEC = 3


def _get_lamp_controller():
    """懒加载台灯控制器：第一次调用时才连接台灯。

    这样即使未安装 python-miio 或未配置 lamp_config.json，
    服务器照常启动，其它（视觉）接口不受影响。
    """
    global _lamp_controller, _lamp_error
    if _lamp_controller is not None:
        return _lamp_controller
    if _lamp_error is not None:
        raise _lamp_error
    try:
        from lamp_controller import MijiaLampController
        _lamp_controller = MijiaLampController()
    except Exception as e:  # 记录错误，避免每次请求都重试
        _lamp_error = e
        raise
    return _lamp_controller


def _run_lamp_action(action, value):
    """Run one lamp action on the dedicated worker thread."""
    ctrl = _get_lamp_controller()
    if action == "get_status":
        return ctrl.get_status()
    if action == "set_power":
        return ctrl.set_power(value)
    if action == "set_brightness":
        return ctrl.set_brightness(value)
    if action == "set_color_temp":
        return ctrl.set_color_temp(value)
    raise ValueError(f"unknown action: {action}")


@app.route("/lamp", methods=["POST"])
def lamp():
    """台灯控制接口。

    请求体 (JSON):
      {"action": "get_status"}
      {"action": "set_power", "value": true}
      {"action": "set_brightness", "value": 60}
      {"action": "set_color_temp", "value": 4000}

    响应体 (JSON):
      {"ok": true, "power": false, "brightness": 1, "color_temp": 5100}
      {"ok": true, "power": true}
      {"ok": false, "error": "..."}

    统一返回 200，成功/失败用 ok 字段区分，便于 ESP32 侧解析错误信息。
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "invalid JSON"})

    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"})

    action = data.get("action", "")
    value = data.get("value")

    if action not in {"get_status", "set_power", "set_brightness", "set_color_temp"}:
        return jsonify({"ok": False, "error": f"unknown action: {action}"})

    global _lamp_future
    if _lamp_future is not None and not _lamp_future.done():
        return jsonify({"ok": False, "error": "lamp request is still in progress"})

    _lamp_future = _lamp_executor.submit(_run_lamp_action, action, value)
    try:
        result = _lamp_future.result(timeout=LAMP_REQUEST_TIMEOUT_SEC)
        return jsonify({"ok": True, **result})
    except concurrent.futures.TimeoutError:
        return jsonify({
            "ok": False,
            "error": f"lamp request timed out after {LAMP_REQUEST_TIMEOUT_SEC} seconds",
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ============================================================
# 模型下载
# ============================================================

def _download(url: str, dest: Path) -> bool:
    import urllib.request
    if dest.exists():
        print(f"[OK] already exists: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    print(f"[Download] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] downloaded: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"[ERROR] download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download(MODEL_FACE_URL, MODEL_FACE)
    _download(MODEL_POSE_URL, MODEL_POSE)
    _download(MODEL_EMOTION_URL, MODEL_EMOTION)


# ============================================================
# 启动入口
# ============================================================

def main():
    global face_detector, emotion_detector, heart_detector, blood_pressure_detector, state

    parser = argparse.ArgumentParser(description="ESP32 Bio-Perception Server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8291, help="Listen port (default: 8291)")
    parser.add_argument("--face-model", default=str(MODEL_FACE), help="YuNet face model path")
    parser.add_argument("--emotion-model", default=str(MODEL_EMOTION), help="Emotion ONNX model path")
    parser.add_argument("--pose-model", default=str(MODEL_POSE), help="YOLOv8-pose model path")
    parser.add_argument("--buffer-size", type=int, default=BUFFER_MAXLEN,
                        help=f"frame_buffer maxlen (default: {BUFFER_MAXLEN})")
    parser.add_argument("--heart-method", default="pos", choices=list(HEART_METHODS),
                        help=f"rPPG 心率信号提取方法 (default: pos, 可选: {', '.join(HEART_METHODS)})")
    parser.add_argument("--bp-method", default="pos", choices=list(HEART_METHODS),
                        help="血压 rPPG 波形提取方法 (default: pos)")
    parser.add_argument("--bp-model", default="", help="rPPG-BP 导出的 ONNX 模型路径（可选）")
    parser.add_argument("--bp-model-config", default="",
                        help="BP 模型 JSON 配置路径（input_length/normalization，可选）")
    parser.add_argument("--bp-window-seconds", type=float, default=7.0,
                        help="血压测量所需连续 rPPG 窗口秒数 (default: 7)")
    parser.add_argument("--bp-strict-quality", action="store_true",
                        help="启用人脸可见率、采样率和 SNR 的严格血压质量门控")
    parser.add_argument("--rppg-max-samples", type=int, default=RPPG_MAXLEN,
                        help=f"紧凑 rPPG 缓冲样本数 (default: {RPPG_MAXLEN})")
    parser.add_argument("--download-model", action="store_true",
                        help="Download ONNX models and exit")
    args = parser.parse_args()

    if args.download_model:
        download_models()
        return

    state = state_mod.ServerState(buffer_maxlen=args.buffer_size,
                                  rppg_maxlen=args.rppg_max_samples)

    # 初始化模型
    print(f"[Init] Loading models...")
    t0 = time.time()

    face_detector = FaceDetector(args.face_model)

    # 情绪模型可选
    if Path(args.emotion_model).exists():
        emotion_detector = EmotionDetector(args.emotion_model)
    else:
        emotion_detector = None
        print(f"[WARN] Emotion model not found: {args.emotion_model} (face_emotion 将仅返回人脸)")

    heart_detector = HeartRateDetector(face_detector, method=args.heart_method)
    blood_pressure_detector = BloodPressureDetector(
        face_detector, method=args.bp_method, model_path=args.bp_model or None,
        config_path=args.bp_model_config or None, window_seconds=args.bp_window_seconds,
        relaxed_quality=not args.bp_strict_quality)
    print(f"[Init] Models loaded in {(time.time() - t0) * 1000:.0f}ms")

    print(f"\n{'='*54}")
    print(f"  ESP32 Bio-Perception Server")
    print(f"  Listen: http://{args.host}:{args.port}")
    print(f"  Tasks : face_emotion / posture / heart_rate / blood_pressure")
    print(f"  rPPG  : {args.heart_method} (可选: {', '.join(HEART_METHODS)})")
    print(f"  Buffer: {args.buffer_size} frames + {args.rppg_max_samples} rPPG samples")
    print(f"  BP    : {args.bp_window_seconds:.0f}s window, quality={'strict' if args.bp_strict_quality else 'relaxed'}, model={'ready' if blood_pressure_detector.model.ready else blood_pressure_detector.model.error}")
    print(f"{'='*54}\n")

    # threaded=True: 每个 HTTP 请求独立线程，避免单个慢任务阻塞其他请求
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
