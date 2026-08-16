"""
3D 坐姿检测器 - 基于 MediaPipe Pose (World Landmarks)  [独立演示脚本]

本文件是独立的 MediaPipe 3D 坐姿检测演示（不作为服务器模块使用）。
服务器端正式坐姿模块为: detectors/pose_detector.py (YOLOv8n-Pose + 临床角度法)。

优势:
1. 利用 MediaPipe 3D 物理空间坐标（Pose World Landmarks），解耦摄像头透视畸变。
2. 无论摄像头摆在正前方、斜侧面 45° 还是侧面，3D 骨骼角度均保持一致。
3. 动态基准校准（Calibration）：抵消个体体型差异与习惯坐姿偏差。
4. EMA 滑动滤波：消除帧间 3D 关键点抖动，防止误报。

依赖安装:
  pip install opencv-python mediapipe numpy

运行:
  python mediapipe_posture_demo.py
  按 C 校准标准坐姿，按 Q 退出。
"""

import math
import time
import cv2
import numpy as np
import mediapipe as mp


class PostureDetector3D:
    """基于 MediaPipe 3D World Landmarks 的生物力学坐姿检测器"""

    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,  # 0: Lite, 1: Full, 2: Heavy
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 neck_thresh_deg: float = 15.0,   # 相比基准，颈部前倾超过该角度判为不良
                 trunk_thresh_deg: float = 12.0):  # 相比基准，躯干倾斜超过该角度判为不良

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.neck_thresh_deg = neck_thresh_deg
        self.trunk_thresh_deg = trunk_thresh_deg

        # 校准状态 (基准角度)
        self.is_calibrated = False
        self.base_neck_angle = 0.0
        self.base_trunk_angle = 0.0

        # 平滑滤波参数 (Exponential Moving Average)
        self.alpha = 0.3
        self.smooth_neck_angle = None
        self.smooth_trunk_angle = None

    @staticmethod
    def _vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个 3D 向量之间的夹角（单位：度）"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _extract_3d_metrics(self, world_landmarks) -> tuple:
        """从 3D World Landmarks 中提取颈部角与躯干角。

        返回: (success, neck_angle, trunk_angle)
        """
        kpts = world_landmarks.landmark

        # MediaPipe Landmark 索引定义
        # 0: nose, 7: left_ear, 8: right_ear, 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
        def get_pt(idx):
            lm = kpts[idx]
            return np.array([lm.x, lm.y, lm.z])

        # 检查关键点可见度与存在性
        required_indices = [11, 12]  # 双肩必须存在
        for idx in required_indices:
            if kpts[idx].visibility < 0.5:
                return False, 0.0, 0.0

        l_sh, r_sh = get_pt(11), get_pt(12)
        sh_center = (l_sh + r_sh) / 2.0  # 肩部中心

        # 头部位置优先选双耳中点，遮挡时选单耳或鼻子
        head_pts = []
        if kpts[7].visibility > 0.5: head_pts.append(get_pt(7))
        if kpts[8].visibility > 0.5: head_pts.append(get_pt(8))
        
        if head_pts:
            head_center = np.mean(head_pts, axis=0)
        elif kpts[0].visibility > 0.5:
            head_center = get_pt(0)
        else:
            return False, 0.0, 0.0

        # 1) 颈前倾角计算: 肩中心->头中点 向量 与 3D 空间真实垂直向上向量 (0, -1, 0) 的夹角
        # 注意: MediaPipe 3D 坐标系中 -Y 代表物理世界向上
        neck_vector = head_center - sh_center
        up_vector = np.array([0.0, -1.0, 0.0])
        raw_neck_angle = self._vector_angle_3d(neck_vector, up_vector)

        # 2) 躯干倾斜角计算: 髋中心->肩中心 向量 与 3D 空间垂直向上向量 (0, -1, 0) 的夹角
        raw_trunk_angle = 0.0
        if kpts[23].visibility > 0.4 and kpts[24].visibility > 0.4:
            l_hip, r_hip = get_pt(23), get_pt(24)
            hip_center = (l_hip + r_hip) / 2.0
            trunk_vector = sh_center - hip_center
            raw_trunk_angle = self._vector_angle_3d(trunk_vector, up_vector)

        return True, raw_neck_angle, raw_trunk_angle

    def calibrate(self, raw_neck: float, raw_trunk: float):
        """记录当前坐姿为基准标准姿态"""
        self.base_neck_angle = raw_neck
        self.base_trunk_angle = raw_trunk
        self.is_calibrated = True
        print(f"[Calibration] 基准已设定: Base Neck={raw_neck:.1f}°, Base Trunk={raw_trunk:.1f}°")

    def detect(self, image_bgr: np.ndarray) -> dict:
        """检测输入图像的坐姿。

        返回:
          {
             "state": "normal" | "bad_posture" | "unknown" | None,
             "reason": str,
             "metrics": dict,
             "2d_landmarks": pose_landmarks, # 供 OpenCV 画图显示
          }
        """
        # 转为 RGB 供 MediaPipe 推理
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_world_landmarks or not results.pose_landmarks:
            return {"state": None, "reason": "no_person", "metrics": {}, "2d_landmarks": None}

        # 从 3D 物理空间坐标提取角度
        success, raw_neck, raw_trunk = self._extract_3d_metrics(results.pose_world_landmarks)

        if not success:
            return {
                "state": "unknown", 
                "reason": "keypoints_missing", 
                "metrics": {}, 
                "2d_landmarks": results.pose_landmarks
            }

        # 指数滑动平均 (EMA) 平滑去噪
        if self.smooth_neck_angle is None:
            self.smooth_neck_angle = raw_neck
            self.smooth_trunk_angle = raw_trunk
        else:
            self.smooth_neck_angle = self.alpha * raw_neck + (1 - self.alpha) * self.smooth_neck_angle
            self.smooth_trunk_angle = self.alpha * raw_trunk + (1 - self.alpha) * self.smooth_trunk_angle

        curr_neck = self.smooth_neck_angle
        curr_trunk = self.smooth_trunk_angle

        metrics = {
            "raw_neck": round(raw_neck, 1),
            "raw_trunk": round(raw_trunk, 1),
            "neck_angle": round(curr_neck, 1),
            "trunk_angle": round(curr_trunk, 1),
            "base_neck": round(self.base_neck_angle, 1),
            "base_trunk": round(self.base_trunk_angle, 1),
            "delta_neck": round(curr_neck - self.base_neck_angle, 1),
            "delta_trunk": round(curr_trunk - self.base_trunk_angle, 1),
        }

        # 规则判断
        if not self.is_calibrated:
            # 未校准时提示需要校准，默认给 normal
            return {
                "state": "normal", 
                "reason": "need_calibration", 
                "metrics": metrics, 
                "2d_landmarks": results.pose_landmarks
            }

        # 计算与标准坐姿基准线的偏差量 (Delta)
        delta_neck = curr_neck - self.base_neck_angle
        delta_trunk = abs(curr_trunk - self.base_trunk_angle)

        if delta_neck > self.neck_thresh_deg:
            state = "bad_posture"
            reason = "head_forward"  # 头前倾 / 低头
        elif delta_trunk > self.trunk_thresh_deg:
            state = "bad_posture"
            reason = "trunk_lean"    # 弯腰 / 驼背 / 严重前倾
        else:
            state = "normal"
            reason = "ok"

        return {
            "state": state, 
            "reason": reason, 
            "metrics": metrics, 
            "2d_landmarks": results.pose_landmarks
        }


# ==================== 实时 Webcam 测试主程序 ====================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        return

    # 初始化 3D 坐姿检测器
    detector = PostureDetector3D(model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    print("\n" + "=" * 50)
    print("【坐姿检测系统已启动 - MediaPipe 3D 版】")
    print(" 使用说明:")
    print("  1. 请坐好保持【正确坐姿】。")
    print("  2. 按键盘 【 C 】 键完成基准坐姿校准。")
    print("  3. 按键盘 【 Q 】 键退出程序。")
    print("=" * 50 + "\n")

    calib_countdown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 图像水平翻转（镜像视效）
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 执行检测
        res = detector.detect(frame)
        state = res["state"]
        reason = res["reason"]
        metrics = res["metrics"]
        landmarks2d = res["2d_landmarks"]

        # 处理校准倒计时逻辑
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            calib_countdown = 10  # 触发校准，持续缓冲 10 帧做平滑

        if calib_countdown > 0:
            calib_countdown -= 1
            if calib_countdown == 0 and "raw_neck" in metrics:
                detector.calibrate(metrics["raw_neck"], metrics["raw_trunk"])

        # ---------- 绘制结果UI ----------

        # 1. 绘制 2D 骨骼骨架
        if landmarks2d:
            mp_drawing.draw_landmarks(
                frame, 
                landmarks2d, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        # 2. 状态标签与颜色设置
        if state == "normal":
            color = (0, 255, 0)     # 绿色
            status_text = "Status: GOOD"
        elif state == "bad_posture":
            color = (0, 0, 255)     # 红色
            status_text = f"Status: BAD ({reason})"
        elif state == "unknown":
            color = (0, 255, 255)   # 黄色
            status_text = "Status: UNKNOWN (Keypoints lost)"
        else:
            color = (255, 255, 255)
            status_text = "No Person Detected"

        # 顶部状态栏背景框
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 显示校准状态提示
        if not detector.is_calibrated:
            cv2.putText(frame, "PLEASE PRESS 'C' TO CALIBRATE YOUR POSTURE", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        else:
            cv2.putText(frame, f"Calibrated | Delta Neck: {metrics.get('delta_neck', 0):+.1f} deg | Delta Trunk: {metrics.get('delta_trunk', 0):+.1f} deg",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 3. 画面左下角绘制详细数据
        if metrics:
            info_y = h - 60
            cv2.putText(frame, f"3D Neck Angle:  {metrics['neck_angle']} deg (Base: {metrics['base_neck']})",
                        (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"3D Trunk Angle: {metrics['trunk_angle']} deg (Base: {metrics['base_trunk']})",
                        (20, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("3D Posture Detector (MediaPipe)", frame)

        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
