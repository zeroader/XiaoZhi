"""
人脸检测器 - YuNet (cv2.FaceDetectorYN)

模型: face_detection_yunet_2023mar.onnx
下载: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
"""

import cv2
import numpy as np

# 检测参数
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000


class FaceDetector:
    """人脸检测器，封装 cv2.FaceDetectorYN（YuNet）"""

    def __init__(self, model_path: str):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
        )
        print(f"[Model] YuNet loaded: {model_path}")

    def detect(self, image_rgb: np.ndarray, src_w: int, src_h: int) -> list:
        """检测人脸，返回 bbox 列表。

        返回: [{ "bbox": {"x","y","width","height"}, "confidence": float }]
        """
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        results = []
        if faces is None:
            return results

        scale_x = src_w / w if w > 0 else 1.0
        scale_y = src_h / h if h > 0 else 1.0

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
                "bbox": {"x": bx, "y": by, "width": bw, "height": bh},
                "confidence": round(score, 4),
            })

        return results

    def detect_primary_face(self, image_rgb: np.ndarray, src_w: int, src_h: int) -> dict:
        """返回置信度最高的人脸（用于情绪/心率等下游任务），无人脸返回 None。"""
        faces = self.detect(image_rgb, src_w, src_h)
        if not faces:
            return None
        return max(faces, key=lambda f: f["confidence"])
