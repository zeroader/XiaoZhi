"""
坐姿检测器 - YOLOv8n Pose（单侧自适应 + 基准校准 + CVA 增强版）

模型: models/yolov8n_pose.onnx
输入: 640x640 RGB
输出: 17 个人体关键点 (COCO)

下载: https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx

改进特性:
  1. 单侧自适应 (Single-Side Adaptive): 侧斜视角遮挡时自动切换使用置信度最高的
     单侧耳/肩/髋点，避免缺失侧拉歪坐标中点。
  2. 基准姿态校准 (Baseline Calibration): 记录标准坐姿作为零点，根据相对角度
     变化量 (Delta) 判断，消除个体体型差异与视角偏差。通过请求字段
     `"calibrate": true` 触发。
  3. CVA (颅颈角) + 躯干倾角: 临床 CVA 模型判断头前倾，配合 EMA 平滑去噪。

输出 state:
  normal       正常
  bad_posture  坐姿不良（head_forward / trunk_lean / head_down）
  unknown      数据不足（关键点缺失），无法评估
  None         未检测到人体
"""

import math

import cv2
import numpy as np

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# 角度判定阈值（度）
# 绝对阈值：未校准时也生效，CVA 是"肩->耳与水平线夹角"，正常坐姿约 85~95°，
#           头前倾时 CVA 变小，低于 CVA_BAD_ANGLE 判定不良
CVA_BAD_ANGLE = 70.0       # CVA 低于该值判定头前倾
TRUNK_BAD_ANGLE = 25.0     # 躯干倾斜角超过该值判定驼背/前倾（绝对）
# 校准后相对偏差阈值（相比基准坐姿的增量，更灵敏）
DELTA_NECK_BAD_ANGLE = 12.0   # 头前倾：CVA 比基准再减小超过该值判不良
DELTA_TRUNK_BAD_ANGLE = 10.0  # 躯干倾斜比基准再增加超过该值判不良

# COCO 17 关键点索引
KP_NOSE = 0
KP_L_EAR = 3
KP_R_EAR = 4
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
    """YOLOv8n-Pose 关键点检测 + 单侧自适应 + 自动/手动校准 + CVA 姿态分析

    校准策略（ESP32 无需手动操作）:
      - 手动校准: 请求带 `"calibrate": true` 立即把当前姿态设为基准（优先级最高）
      - 自动校准: 连续 AUTO_CALIB_WARMUP 帧后，用滑动窗口角度中位数作为个人
        习惯坐姿基准（无需 ESP32 任何操作）；绝对阈值始终兜底
    """

    def __init__(self, model_path: str, input_size: int = INPUT_SIZE,
                 conf_threshold: float = CONF_THRESHOLD,
                 nms_threshold: float = NMS_THRESHOLD):
        load = getattr(cv2.dnn, "readNetFromONNX", None) or cv2.dnn.readNetFromOnnx
        self.net = load(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 校准状态 (基准角度)
        self.is_calibrated = False
        self.base_cva = 0.0
        self.base_trunk_angle = 0.0
        self._manual_calib = False     # 手动校准优先于自动校准

        # 自动校准: 滑动窗口缓存 (cva, trunk)，中位数作为习惯坐姿基准
        self.auto_win = []
        self.AUTO_CALIB_WARMUP = 20      # 至少累积多少帧才开始自动校准
        self.AUTO_CALIB_WINDOW = 60      # 滑动窗口大小

        # 平滑滤波 (Exponential Moving Average)
        self.alpha = 0.3
        self.smooth_cva = None
        self.smooth_trunk_angle = None

        print(f"[Model] YOLOv8-Pose loaded: {model_path}")

    # ---------- 校准 API ----------

    def calibrate(self, raw_cva: float, raw_trunk: float):
        """手动校准：记录当前 CVA 与躯干角度作为基准坐姿（标准零点）"""
        self.base_cva = raw_cva
        self.base_trunk_angle = raw_trunk
        self.is_calibrated = True
        self._manual_calib = True
        print(f"[Calibration] 手动基准已设定: Base CVA={raw_cva:.1f}°, Base Trunk={raw_trunk:.1f}°")

    def _auto_calibrate(self, raw_cva: float, raw_trunk: float):
        """自动校准：累积滑动窗口，用"最直"样本（CVA 较大的后 1/3 中位数）
        作为个人习惯坐姿基准。ESP32 无需任何操作即可逐步建立基准。"""
        self.auto_win.append((raw_cva, raw_trunk))
        if len(self.auto_win) > self.AUTO_CALIB_WINDOW:
            self.auto_win.pop(0)

        if len(self.auto_win) < self.AUTO_CALIB_WARMUP:
            return False

        # 取 CVA 最大的 1/3 样本（即最接近笔直坐姿的时刻）求中位数
        cvas = sorted(v[0] for v in self.auto_win)
        n_good = max(1, len(cvas) // 3)
        good = cvas[-n_good:]
        med_cva = good[len(good) // 2]

        # 躯干角同样取"最直"（最小）的样本中位数
        trunks = sorted(v[1] for v in self.auto_win)
        good_t = trunks[:n_good]
        med_trunk = good_t[len(good_t) // 2]

        self.base_cva = med_cva
        self.base_trunk_angle = med_trunk
        self.is_calibrated = True
        return True

    def reset_calibration(self):
        """清空校准状态（重启校准流程）"""
        self.is_calibrated = False
        self.base_cva = 0.0
        self.base_trunk_angle = 0.0
        self.auto_win.clear()
        print("[Calibration] 已重置")

    # ---------- 推理与后处理 ----------

    def _postprocess(self, out: np.ndarray, orig_shape: tuple) -> list:
        """YOLOv8 pose 输出 (1, 56, 8400) -> 关键点结果列表。"""
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
        """检测人体姿态并进行鲁棒坐姿评估。

        返回:
          {"state": "normal"|"bad_posture"|"unknown"|None, "persons": [...],
           "reason": str, "metrics": {...}}
          - state=None    : 未检测到人体 (reason=no_person)
          - state=unknown : 关键点数据不足，无法评估 (reason=keypoints_missing)
        """
        h, w = image_rgb.shape[:2]
        canvas, r, pad_x, pad_y = letterbox(image_rgb, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size),
                                     (0, 0, 0), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        persons = self._postprocess(out, (h, w))

        if not persons:
            return {"state": None, "persons": [], "reason": "no_person", "metrics": {}}

        # 选取置信度最高的人做姿态分析
        primary = max(persons, key=lambda p: p["confidence"])
        kpts = {k["name"]: k for k in primary["keypoints"]}
        state, reason, metrics = self._analyze_posture(kpts)

        return {"state": state, "reason": reason, "metrics": metrics,
                "persons": persons, "analyzed_person": primary}

    # ---------- 几何辅助与单侧自适应 ----------

    @staticmethod
    def _angle_deg(px: float, py: float, qx: float, qy: float,
                   ref_x: float, ref_y: float) -> float:
        """向量 P->Q 与参考向量 ref 的夹角（度）。"""
        vx, vy = qx - px, qy - py
        ln = math.hypot(vx, vy)
        if ln < 1e-6:
            return 0.0
        rn = math.hypot(ref_x, ref_y)
        if rn < 1e-6:
            return 0.0
        cos = (vx * ref_x + vy * ref_y) / (ln * rn)
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))

    @staticmethod
    def _get_best_point(kpts: dict, left_name: str, right_name: str,
                        conf_thresh: float = 0.3) -> tuple:
        """【单侧自适应】侧斜视角下优先选清晰单侧点，避免缺失侧拉歪中点。

        返回: ((x, y), is_valid)
        """
        lk = kpts.get(left_name)
        rk = kpts.get(right_name)

        l_ok = lk is not None and lk["confidence"] > conf_thresh
        r_ok = rk is not None and rk["confidence"] > conf_thresh

        if l_ok and r_ok:
            return ((lk["x"] + rk["x"]) / 2.0, (lk["y"] + rk["y"]) / 2.0), True
        elif l_ok:
            return (lk["x"], lk["y"]), True
        elif r_ok:
            return (rk["x"], rk["y"]), True
        return (0.0, 0.0), False

    # ---------- 姿态分析 (CVA + 基准校准) ----------

    def _analyze_posture(self, kpts: dict) -> tuple:
        """基于单侧自适应关键点、CVA 模型与动态校准判断坐姿。

        返回: (state, reason, metrics)
        """
        ear_pt, ear_valid = self._get_best_point(kpts, "left_ear", "right_ear")
        shoulder_pt, sh_valid = self._get_best_point(kpts, "left_shoulder", "right_shoulder")
        hip_pt, hip_valid = self._get_best_point(kpts, "left_hip", "right_hip")
        nose = kpts.get("nose")

        # 降级 fallback: 耳朵不可用时用鼻子代替头位
        if not ear_valid and nose is not None and nose["confidence"] > 0.3:
            ear_pt = (nose["x"], nose["y"])
            ear_valid = True

        # 数据完整性检查: 肩/头缺失（如离摄像头过近）返回 unknown
        if not sh_valid or not ear_valid:
            return "unknown", "keypoints_missing", {}

        sh_x, sh_y = shoulder_pt
        ear_x, ear_y = ear_pt

        # 1) CVA 颅颈角: 肩->耳连线 与 过肩膀水平向右向量 (1,0) 的夹角
        #    正常坐姿耳在肩正上方(约90°)，头前倾时 CVA 变小
        cva_raw = self._angle_deg(sh_x, sh_y, ear_x, ear_y, 1.0, 0.0)

        # 2) 躯干倾斜角: 肩->髋连线 与 垂直向下向量 (0,1) 的夹角
        trunk_raw = 0.0
        if hip_valid:
            hip_x, hip_y = hip_pt
            trunk_raw = self._angle_deg(sh_x, sh_y, hip_x, hip_y, 0.0, 1.0)

        # 3) EMA 平滑去噪
        if self.smooth_cva is None:
            self.smooth_cva = cva_raw
            self.smooth_trunk_angle = trunk_raw
        else:
            self.smooth_cva = self.alpha * cva_raw + (1 - self.alpha) * self.smooth_cva
            self.smooth_trunk_angle = self.alpha * trunk_raw + (1 - self.alpha) * self.smooth_trunk_angle

        curr_cva = self.smooth_cva
        curr_trunk = self.smooth_trunk_angle

        # 4) 相对基准的变化量 Delta
        delta_cva = self.base_cva - curr_cva      # CVA 变小代表头前倾/低头
        delta_trunk = abs(curr_trunk - self.base_trunk_angle)

        # 4b) 自动校准（手动校准后不再覆盖）
        if not self._manual_calib:
            self._auto_calibrate(cva_raw, trunk_raw)

        metrics = {
            "raw_cva": round(cva_raw, 1),
            "raw_trunk": round(trunk_raw, 1),
            "cva": round(curr_cva, 1),
            "trunk_angle": round(curr_trunk, 1),
            "base_cva": round(self.base_cva, 1),
            "base_trunk": round(self.base_trunk_angle, 1),
            "delta_cva": round(delta_cva, 1),
            "delta_trunk": round(delta_trunk, 1),
            "calibrated": self.is_calibrated,
        }

        # 5) 趴桌/低头特例: 头部高度低于肩膀
        head_down = (ear_y > sh_y)

        # 综合判定（绝对阈值始终生效，校准后叠加 Delta 灵敏度）:
        if head_down:
            return "bad_posture", "head_down", metrics
        # CVA 过小 = 头前倾/低头（绝对）；校准后若比基准再减小更多也判不良
        if curr_cva < CVA_BAD_ANGLE or (self.is_calibrated and delta_cva > DELTA_NECK_BAD_ANGLE):
            return "bad_posture", "head_forward", metrics
        # 躯干倾斜（绝对）；校准后相对增量也判不良
        if curr_trunk > TRUNK_BAD_ANGLE or (self.is_calibrated and delta_trunk > DELTA_TRUNK_BAD_ANGLE):
            return "bad_posture", "trunk_lean", metrics

        return "normal", "ok", metrics
