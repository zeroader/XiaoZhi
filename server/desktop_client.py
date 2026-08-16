"""
PC 桌面端测试客户端（模拟 ESP32 的调度行为）

功能:
  - 调用本机摄像头 (默认 index 0)
  - 按协议把 JPEG 发给生物感知服务器 (POST /detect)
  - cv2.imshow 叠加显示: 人脸框 + 情绪标签 / pose 关键点骨架 / 心率数值 / 姿态状态

调度（由本客户端控制频率，服务器不做任何定时任务）:
  - face_emotion  每帧
  - posture       每 5 秒
  - heart_rate    每 10 秒

用法:
  python desktop_client.py --url http://127.0.0.1:8291/detect --camera 0
  按 q 退出
"""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

# ============================================================
# 常量
# ============================================================

# COCO 17 关键点骨架连接 (idx pairs)
POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 脸
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 手臂
    (5, 11), (6, 12), (11, 12),              # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]

POSTURE_INTERVAL = 5.0    # 坐姿请求间隔（秒）
HEART_INTERVAL = 10.0     # 心率请求间隔（秒）
WORK_WIDTH = 640          # 发送/显示的工作宽度（等比缩放，控制带宽）
JPEG_QUALITY = 80

# 颜色 BGR
COLOR_FACE = (0, 0, 255)      # 人脸框红
COLOR_EMOTION = (0, 255, 255) # 情绪文字黄
COLOR_POSE = (0, 255, 0)      # pose 骨架绿
COLOR_HR = (255, 0, 0)        # 心率文字蓝
COLOR_POSTURE = (255, 255, 0) # 姿态状态青


# ============================================================
# 工具
# ============================================================

def encode_jpeg(frame: np.ndarray) -> str:
    """BGR frame -> base64 JPEG 字符串"""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def post_detect(url: str, frame_id: int, task: str, b64: str, w: int, h: int,
                calibrate: bool = False) -> dict:
    """POST /detect，返回解析后的 JSON"""
    payload = {
        "frame_id": frame_id,
        "task": task,
        "image": {"data": b64, "width": w, "height": h, "format": "jpeg"},
    }
    if calibrate:
        payload["calibrate"] = True
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ============================================================
# 绘制
# ============================================================

def draw_face_emotion(img: np.ndarray, result: dict):
    """人脸框 + 情绪标签"""
    face = result.get("face")
    if not face:
        return
    b = face["bbox"]
    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_FACE, 2)

    emotion = result.get("emotion")
    label = "no_face"
    if emotion and "label" in emotion:
        label = f"{emotion['label']} {emotion.get('confidence', 0):.2f}"
    cv2.putText(img, label, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EMOTION, 1, cv2.LINE_AA)


def draw_pose(img: np.ndarray, result: dict):
    """pose 关键点 + 骨架"""
    persons = result.get("persons") or []
    for p in persons:
        kpts = p["keypoints"]
        # 关键点
        for k in kpts:
            if k["confidence"] > 0.3:
                cv2.circle(img, (k["x"], k["y"]), 3, COLOR_POSE, -1, cv2.LINE_AA)
        # 骨架连线
        for a, b in POSE_SKELETON:
            pa, pb = kpts[a], kpts[b]
            if pa["confidence"] > 0.3 and pb["confidence"] > 0.3:
                cv2.line(img, (pa["x"], pa["y"]), (pb["x"], pb["y"]),
                         COLOR_POSE, 2, cv2.LINE_AA)


def draw_heart_rate(img: np.ndarray, hr: dict):
    """心率数值（左上角）"""
    if not hr or "bpm" not in hr:
        text = "HR: --"
    else:
        text = f"HR: {hr['bpm']} BPM ({hr.get('confidence', 0):.2f})"
    cv2.putText(img, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_HR, 2, cv2.LINE_AA)


def draw_posture_state(img: np.ndarray, posture: dict):
    """姿态状态（左上角第二行）"""
    if not posture:
        text = "posture: --"
    elif posture.get("state") is None:
        text = "posture: no_person"
    elif posture["state"] == "unknown":
        text = "posture: unknown (离远一点)"   # 关键点不足，如肩膀出画面
    else:
        state = posture["state"]
        reason = posture.get("reason", "")
        text = f"posture: {state}"
        if reason:
            text += f" ({reason})"
        metrics = posture.get("metrics") or {}
        if metrics:
            text += f" dCVA={metrics.get('delta_cva')} dTrunk={metrics.get('delta_trunk')}"
            if not metrics.get("calibrated"):
                text += " [未校准,按C]"
    cv2.putText(img, text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_POSTURE, 2, cv2.LINE_AA)


# ============================================================
# 主循环
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PC 桌面端生物感知测试客户端")
    parser.add_argument("--url", default="http://127.0.0.1:8291/detect",
                        help="服务器 /detect 地址 (default: http://127.0.0.1:8291/detect)")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 index (default: 0)")
    parser.add_argument("--width", type=int, default=WORK_WIDTH, help="发送宽度 (default: 640)")
    parser.add_argument("--fps", type=float, default=15.0, help="face_emotion 目标频率 (default: 15)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {args.camera}")
        return

    frame_id = 0
    last_posture_t = 0.0
    last_heart_t = 0.0
    last_posture = None
    last_hr = None
    min_interval = 1.0 / args.fps

    print(f"[Init] 连接服务器: {args.url}")
    print("[Init] 按 q 退出 | C 校准坐姿 | face_emotion 每帧 / posture 每5s / heart_rate 每10s\n")

    # 记录待校准标志（按 C 后在下一次 posture 请求中触发）
    pending_calibrate = False

    while True:
        t_loop = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 读取摄像头帧失败")
            break

        # 等比缩放到工作宽度（控制带宽 + 保证绘制坐标系一致）
        h, w = frame.shape[:2]
        if w != args.width:
            frame = cv2.resize(frame, (args.width, int(h * args.width / w)))
        h, w = frame.shape[:2]

        b64 = encode_jpeg(frame)
        frame_id += 1

        # 1) face_emotion: 每帧
        try:
            res = post_detect(args.url, frame_id, "face_emotion", b64, w, h)
            draw_face_emotion(frame, res.get("result") or {})
        except (urllib.error.URLError, OSError) as e:
            print(f"[ERROR] face_emotion 请求失败: {e}")
            break

        # 2) posture: 每 5s（或按键 C 触发校准）
        now = time.time()
        if pending_calibrate or now - last_posture_t >= POSTURE_INTERVAL:
            try:
                res = post_detect(args.url, frame_id, "posture", b64, w, h,
                                  calibrate=pending_calibrate)
                last_posture = res.get("result") or {}
                last_posture_t = now
                if pending_calibrate:
                    print(f"[Calibrate] {res.get('result', {}).get('reason', '')} "
                          f"{res.get('result', {}).get('metrics', {})}")
                    pending_calibrate = False
            except (urllib.error.URLError, OSError) as e:
                print(f"[ERROR] posture 请求失败: {e}")

        # 3) heart_rate: 每 10s
        if now - last_heart_t >= HEART_INTERVAL:
            try:
                res = post_detect(args.url, frame_id, "heart_rate", b64, w, h)
                last_hr = res.get("result") or {}
                last_heart_t = now
            except (urllib.error.URLError, OSError) as e:
                print(f"[ERROR] heart_rate 请求失败: {e}")

        # 叠加绘制
        draw_pose(frame, last_posture or {})
        draw_heart_rate(frame, last_hr)
        draw_posture_state(frame, last_posture)

        cv2.imshow("Bio-Perception Client", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") or key == ord("C"):
            pending_calibrate = True

        # 节流到目标频率
        elapsed = time.time() - t_loop
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
