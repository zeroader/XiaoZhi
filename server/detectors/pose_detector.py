"""
坐姿检测器 - YOLOv8n Pose

模型: models/yolov8n_pose.onnx
输入: 640x640 RGB
输出: 17 个人体关键点 (COCO)

下载: https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx

姿态判断（规则）:
  - normal      头部与肩膀关系正常
  - bad_posture 驼背（头前倾）或趴桌（头部高度过低）
"""

import cv2
import numpy as np

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# COCO 17 关键点索引
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def letterbox(img: np.ndarray, new_shape: int) -> tuple:
    """等比缩放 + 灰边填充，返回 (新图, 比例, pad_x, pad_y)。"""
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


class PoseDetector:
    """YOLOv8n-Pose 关键点检测 + 姿态规则分析"""

    def __init__(self, model_path: str, input_size: int = INPUT_SIZE,
                 conf_threshold: float = CONF_THRESHOLD,
                 nms_threshold: float = NMS_THRESHOLD):
        load = getattr(cv2.dnn, "readNetFromONNX", None) or cv2.dnn.readNetFromOnnx
        self.net = load(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        print(f"[Model] YOLOv8-Pose loaded: {model_path}")

    # ---------- 推理 ----------

    def _postprocess(self, out: np.ndarray, orig_shape: tuple) -> list:
        """YOLOv8 pose 输出 (1, 56, 8400) -> 关键点结果列表。"""
        # 转置为 (8400, 56): 4 bbox + 1 conf + 51 kpt(17*3)
        pred = np.transpose(out[0], (1, 0))   # (8400, 56)
        boxes_xywh = pred[:, :4]
        conf = pred[:, 4]
        kpts = pred[:, 5:].reshape(-1, 17, 3)  # (8400, 17, 3) x,y,visibility

        keep = conf > self.conf_threshold
        if not keep.any():
            return []

        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        kpts = kpts[keep]

        # cx,cy,w,h -> x1,y1,x2,y2
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf_threshold,
                                self.nms_threshold)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).reshape(-1)

        h, w = orig_shape[:2]
        r = self.input_size / max(h, w)
        pad_x = (self.input_size - w * r) / 2
        pad_y = (self.input_size - h * r) / 2

        results = []
        for i in idxs:
            bx1, by1, bx2, by2 = boxes[i]
            # 映射回原始坐标
            ox1 = int(round((bx1 - pad_x) / r))
            oy1 = int(round((by1 - pad_y) / r))
            ox2 = int(round((bx2 - pad_x) / r))
            oy2 = int(round((by2 - pad_y) / r))
            ox1 = max(0, min(ox1, w - 1)); oy1 = max(0, min(oy1, h - 1))
            ox2 = max(0, min(ox2, w - 1)); oy2 = max(0, min(oy2, h - 1))

            kp = []
            for j in range(17):
                kx = round(float((kpts[i, j, 0] - pad_x) / r))
                ky = round(float((kpts[i, j, 1] - pad_y) / r))
                kc = float(kpts[i, j, 2])
                kp.append({"name": KP_NAMES[j], "x": kx, "y": ky, "confidence": round(kc, 3)})

            results.append({
                "bbox": {"x": ox1, "y": oy1, "width": ox2 - ox1, "height": oy2 - oy1},
                "confidence": round(float(conf[i]), 4),
                "keypoints": kp,
            })
        return results

    def detect(self, image_rgb: np.ndarray) -> dict:
        """检测人体姿态。

        返回:
          {"state": "normal"|"bad_posture", "persons": [...], "reason": str}
          未检测到人体时 persons 为空, state 为 None。
        """
        h, w = image_rgb.shape[:2]
        canvas, r, pad_x, pad_y = letterbox(image_rgb, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size),
                                     (0, 0, 0), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        persons = self._postprocess(out, (h, w))

        if not persons:
            return {"state": None, "persons": [], "reason": "no_person"}

        # 选取置信度最高的人做姿态分析
        primary = max(persons, key=lambda p: p["confidence"])
        kpts = {k["name"]: k for k in primary["keypoints"]}
        state, reason = self._analyze_posture(kpts)

        return {"state": state, "reason": reason,
                "persons": persons, "analyzed_person": primary}

    # ---------- 姿态规则 ----------

    def _analyze_posture(self, kpts: dict) -> tuple:
        """基于 nose / shoulder / hip 判断坐姿。

        返回: (state, reason)
        """
        nose = kpts.get("nose")
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        lh = kpts.get("left_hip")
        rh = kpts.get("right_hip")

        # 关键点置信度过滤
        def usable(k):
            return k is not None and k["confidence"] > 0.3

        if not usable(nose) or not usable(ls) or not usable(rs):
            return "bad_posture", "keypoints_missing"

        shoulder_cx = (ls["x"] + rs["x"]) / 2.0
        shoulder_cy = (ls["y"] + rs["y"]) / 2.0
        shoulder_w = max(1.0, abs(rs["x"] - ls["x"]))

        # 1) 驼背：头前倾（nose 明显偏离肩膀中心）
        nose_offset = (nose["x"] - shoulder_cx) / shoulder_w
        if abs(nose_offset) > 0.45:
            return "bad_posture", "head_forward"

        # 2) 趴桌/低头：头部高度接近甚至低于肩膀
        #    正常坐姿 nose 应在 shoulder 上方一定距离
        if usable(lh) and usable(rh):
            hip_cy = (lh["y"] + rh["y"]) / 2.0
            torso_h = max(1.0, hip_cy - shoulder_cy)
            drop = (nose["y"] - shoulder_cy) / torso_h  # >0 表示头部在肩膀下方
            if drop > 0.25:
                return "bad_posture", "head_down"

        return "normal", "ok"
