"""
ESP32 在线人脸检测服务器
接收 JSON+Base64 图片，使用 ONNX 模型做推理，返回 bbox 结果

协议:
  Request  POST /detect
  {
    "type": "face",
    "images": ["base64_encoded_jpeg"],
    "width": 320,
    "height": 240,
    "format": "jpeg"
  }

  Response
  {
    "detections": [
      {
        "class": "face",
        "confidence": 0.95,
        "bbox": {"x": 10, "y": 20, "width": 100, "height": 150}
      }
    ]
  }

启动方式:
  1. 安装依赖: pip install -r server_requirements.txt
  2. 下载模型: python face_detect_server.py --download-model
  3. 启动服务: python face_detect_server.py --host 0.0.0.0 --port 5000
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2

app = Flask(__name__)

# ============================================================
# 模型配置
# ============================================================

# YuNet face detection model from OpenCV Zoo
# Uses cv2.FaceDetectorYN (built-in OpenCV wrapper that handles pre/post-processing)
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_FILE = "face_detection_yunet_2023mar.onnx"

# 检测参数
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000


# ============================================================
# ONNX 推理引擎
# ============================================================

class FaceDetector:
    """人脸检测器，封装 cv2.FaceDetectorYN（YuNet）"""

    def __init__(self, model_path: str):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
        )
        print(f"[Model] YuNet loaded: {model_path}")

    def detect(self, image_rgb: np.ndarray, src_w: int, src_h: int) -> list:
        """检测人脸，返回 bbox 列表"""
        # YuNet 内部处理 resize/预处理/后处理，直接传 BGR
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        results = []
        if faces is None:
            return results

        scale_x = src_w / w
        scale_y = src_h / h

        for face in faces:
            # face format: [x, y, w, h, ...keypoints x10, score]
            bx = int(float(face[0]) * scale_x)
            by = int(float(face[1]) * scale_y)
            bw = int(float(face[2]) * scale_x)
            bh = int(float(face[3]) * scale_y)
            score = float(face[-1])

            if score < SCORE_THRESHOLD:
                continue

            # 边界裁剪
            if bx < 0:
                bw += bx
                bx = 0
            if by < 0:
                bh += by
                by = 0
            if bx + bw > src_w:
                bw = src_w - bx
            if by + bh > src_h:
                bh = src_h - by

            if bw <= 0 or bh <= 0:
                continue

            results.append({
                "class": "face",
                "confidence": round(score, 4),
                "bbox": {"x": bx, "y": by, "width": bw, "height": bh}
            })

        return results


# ============================================================
# 全局检测器实例
# ============================================================

detector: FaceDetector = None
_last_frame_time = None  # 上一帧到达的时间戳，用于计算帧间隔和 FPS
_last_response_time = None  # 上一次响应发出的时间戳，用于估算网络+客户端耗时


# ============================================================
# HTTP API
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():
    """人脸检测 API"""
    global _last_frame_time, _last_response_time
    t_start = time.time()

    # 客户端(ESP32)+网络耗时：上一次响应发出后到本次请求到达
    client_net_ms = 0.0
    if _last_response_time is not None:
        client_net_ms = (t_start - _last_response_time) * 1000

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    # 校验请求
    images_b64 = data.get("images", [])
    if not images_b64 or not isinstance(images_b64, list):
        return jsonify({"error": "`images` must be a non-empty list"}), 400

    src_width = data.get("width", 0)
    src_height = data.get("height", 0)

    all_detections = []

    # 统计各阶段耗时（解码 / 推理）与请求体大小
    t_decode_ms = 0.0
    t_infer_ms = 0.0
    payload_kb = 0.0

    for idx, img_b64 in enumerate(images_b64):
        if not img_b64:
            continue

        payload_kb += len(img_b64) / 1024.0

        t0 = time.time()
        try:
            # Base64 解码 -> JPEG bytes
            jpeg_bytes = base64.b64decode(img_b64)
        except Exception:
            return jsonify({"error": f"image[{idx}]: invalid base64"}), 400

        try:
            # JPEG -> RGB numpy array
            image = Image.open(io.BytesIO(jpeg_bytes))
            image_rgb = np.array(image.convert("RGB"))
        except Exception as e:
            return jsonify({"error": f"image[{idx}]: decode failed - {str(e)}"}), 400
        t1 = time.time()

        # 检测
        dets = detector.detect(image_rgb, src_width, src_height)
        t2 = time.time()

        all_detections.extend(dets)
        t_decode_ms += (t1 - t0) * 1000
        t_infer_ms += (t2 - t1) * 1000

    # 计算帧间隔并换算 FPS
    now = time.time()
    interval_ms = 0.0
    fps = 0.0
    if _last_frame_time is not None:
        interval_ms = (now - _last_frame_time) * 1000
        fps = 1000.0 / interval_ms if interval_ms > 0 else 0.0
    _last_frame_time = now
    _last_response_time = now

    t_elapsed = (time.time() - t_start) * 1000
    print(f"[{time.strftime('%H:%M:%S')}] interval={interval_ms:.0f}ms FPS={fps:.1f} | "
          f"client+net={client_net_ms:.0f}ms server={t_elapsed:.0f}ms "
          f"(decode={t_decode_ms:.0f}ms infer={t_infer_ms:.0f}ms) "
          f"payload={payload_kb:.1f}KB detections={len(all_detections)}")

    return jsonify({"detections": all_detections})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_FILE})


# ============================================================
# 模型下载
# ============================================================

def download_model():
    """下载 ONNX 模型"""
    import urllib.request

    model_path = Path(MODEL_FILE)
    if model_path.exists():
        print(f"[OK] Model already exists: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        return

    print(f"[Download] {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[OK] Downloaded: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        sys.exit(1)


# ============================================================
# 启动入口
# ============================================================

def main():
    global detector, SCORE_THRESHOLD

    parser = argparse.ArgumentParser(description="ESP32 Online Face Detection Server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8291, help="Listen port (default: 8291)")
    parser.add_argument("--model", default=MODEL_FILE, help="ONNX model file path")
    parser.add_argument("--download-model", action="store_true", help="Download ONNX model and exit")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Detection score threshold (default: {SCORE_THRESHOLD})")
    args = parser.parse_args()

    if args.download_model:
        download_model()
        return

    # 初始化模型
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print(f"  Run `python {sys.argv[0]} --download-model` to download")
        sys.exit(1)

    SCORE_THRESHOLD = args.threshold

    print(f"[Init] Loading model: {args.model}")
    t0 = time.time()
    detector = FaceDetector(args.model)
    print(f"[Init] Model loaded in {(time.time() - t0) * 1000:.0f}ms")

    print(f"\n{'='*50}")
    print(f"  Face Detection Server")
    print(f"  Listen: http://{args.host}:{args.port}")
    print(f"  Model:  {args.model}")
    print(f"  Score threshold: {SCORE_THRESHOLD}")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
