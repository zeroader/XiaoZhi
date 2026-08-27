"""使用人脸框几何与时序稳定性的轻量二分类坐姿检测器。"""
from collections import deque
from statistics import median
import numpy as np


class FacePostureDetector:
    # 人脸框阈值：相对画面宽度的比例
    MAX_HORIZONTAL_OFFSET = 0.14
    MAX_SWAY_SPAN = 0.20
    MAX_FACE_SCALE_CHANGE = 0.45

    def __init__(self, face_detector, pose_detector=None, warmup_frames=8):
        self.face_detector = face_detector
        self.pose_detector = pose_detector
        self.warmup_frames = warmup_frames
        self.samples = deque(maxlen=60)
        self.motion = deque(maxlen=10)
        self.base_cx = self.base_cy = self.base_w = None
        self.ema_cx = self.ema_w = None
        self.missing_frames = 0
        self.last_state = "normal"

    def reset_calibration(self):
        self.samples.clear(); self.motion.clear()
        self.base_cx = self.base_cy = self.base_w = None
        self.ema_cx = self.ema_w = None
        self.missing_frames = 0; self.last_state = "normal"

    def calibrate(self, metrics=None):
        if not self.samples:
            return False
        self.base_cx = median(v[0] for v in self.samples)
        self.base_cy = median(v[1] for v in self.samples)
        self.base_w = median(v[2] for v in self.samples)
        return True

    def detect(self, image_rgb: np.ndarray) -> dict:
        h, w = image_rgb.shape[:2]
        faces = self.face_detector.detect(image_rgb, w, h)
        if not faces:
            self.missing_frames += 1
            bad = self.missing_frames >= 3
            self.last_state = "bad_posture" if bad else self.last_state
            return {"state": self.last_state,
                    "reason": "face_missing" if bad else "face_temporarily_missing",
                    "metrics": {"missing_frames": self.missing_frames,
                                 "calibrated": self.base_cx is not None}, "persons": []}

        self.missing_frames = 0
        face = max(faces, key=lambda f: f["confidence"])
        box = face["bbox"]
        cx = (box["x"] + box["width"] * .5) / max(w, 1)
        cy = (box["y"] + box["height"] * .5) / max(h, 1)
        fw = box["width"] / max(w, 1)
        if self.ema_cx is None:
            self.ema_cx, self.ema_w = cx, fw
        else:
            self.ema_cx = .35 * cx + .65 * self.ema_cx
            self.ema_w = .35 * fw + .65 * self.ema_w
        self.samples.append((self.ema_cx, cy, self.ema_w))

        if self.base_cx is None and len(self.samples) >= self.warmup_frames:
            good = list(self.samples)[-self.warmup_frames:]
            self.base_cx = median(v[0] for v in good)
            self.base_cy = median(v[1] for v in good)
            self.base_w = median(v[2] for v in good)
        if self.base_cx is None:
            return {"state": "normal", "reason": "calibrating",
                    "metrics": {"calibrated": False}, "persons": faces,
                    "analyzed_person": face}

        offset = self.ema_cx - self.base_cx
        self.motion.append(self.ema_cx)
        span = max(self.motion) - min(self.motion)
        scale_change = abs(self.ema_w - self.base_w) / max(self.base_w, 1e-6)
        sway = (abs(offset) > self.MAX_HORIZONTAL_OFFSET or
                span > self.MAX_SWAY_SPAN)
        face_bad = sway or scale_change > self.MAX_FACE_SCALE_CHANGE
        pose_result = None
        pose_bad = False
        if self.pose_detector is not None:
            pose_result = self.pose_detector.detect(image_rgb)
            pose_bad = pose_result.get("state") == "bad_posture"
        bad = face_bad or pose_bad
        self.last_state = "bad_posture" if bad else "normal"
        if pose_bad and face_bad:
            reason = "face_sway_and_pose"
        elif pose_bad:
            reason = "pose_keypoints"
        elif sway:
            reason = "excessive_sway"
        elif face_bad:
            reason = "distance_change"
        else:
            reason = "ok"
        return {"state": self.last_state,
                "reason": reason,
                "metrics": {"face_center_x": round(self.ema_cx, 4),
                            "baseline_center_x": round(self.base_cx, 4),
                            "horizontal_offset": round(offset, 4),
                            "sway_span": round(span, 4),
                            "face_scale_change": round(scale_change, 4),
                            "face_bad": face_bad,
                            "pose_bad": pose_bad,
                            "calibrated": True},
                "persons": faces, "analyzed_person": face}
