"""
ESP32 生物特征感知服务器 - 全局缓存状态

服务器不做任何定时任务，只负责：
  1. 接收图片
  2. 根据 task 执行推理
  3. 保存最近 N 帧（frame_buffer），供时序分析（如 rPPG 心率）
  4. 返回结果
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple


@dataclass
class FrameRecord:
    """缓存的一帧"""
    frame_id: int
    timestamp: float          # 接收时刻（unix 秒），用于计算采样率
    image: Any                # RGB numpy 数组（已降采样，节省内存）
    width: int = 0            # 原始宽度
    height: int = 0           # 原始高度


class ServerState:
    """全局状态：最近帧缓存 + 最新各任务结果"""

    def __init__(self, buffer_maxlen: int = 30):
        self.buffer_maxlen = buffer_maxlen
        self._lock = threading.RLock()
        self.frame_buffer: Deque[FrameRecord] = deque(maxlen=buffer_maxlen)
        self.latest_face: Optional[dict] = None
        self.latest_emotion: Optional[dict] = None
        self.latest_posture: Optional[dict] = None
        self.latest_heart_rate: Optional[dict] = None
        self._frame_seq = 0

    # ---------- 帧缓存 ----------

    def push_frame(self, image, width: int = 0, height: int = 0) -> int:
        """保存当前帧到缓存，返回自增 frame_id。"""
        with self._lock:
            self._frame_seq += 1
            rec = FrameRecord(
                frame_id=self._frame_seq,
                timestamp=time.time(),
                image=image,
                width=width,
                height=height,
            )
            self.frame_buffer.append(rec)
            return rec.frame_id

    def get_frames(self) -> List[FrameRecord]:
        """返回缓存帧快照（旧->新）。"""
        with self._lock:
            return list(self.frame_buffer)

    def get_recent_frames(self, n: Optional[int] = None) -> List[FrameRecord]:
        """返回最近 n 帧（旧->新），默认全部。"""
        frames = self.get_frames()
        if n is None or n <= 0:
            return frames
        return frames[-n:]

    # ---------- 最新结果 ----------

    def set_face(self, result: dict):
        with self._lock:
            self.latest_face = result

    def set_emotion(self, result: dict):
        with self._lock:
            self.latest_emotion = result

    def set_posture(self, result: dict):
        with self._lock:
            self.latest_posture = result

    def set_heart_rate(self, result: dict):
        with self._lock:
            self.latest_heart_rate = result

    def snapshot(self) -> dict:
        """汇总最新结果，供 /state 查询。"""
        with self._lock:
            return {
                "buffer_frames": len(self.frame_buffer),
                "buffer_maxlen": self.buffer_maxlen,
                "latest_face": self.latest_face,
                "latest_emotion": self.latest_emotion,
                "latest_posture": self.latest_posture,
                "latest_heart_rate": self.latest_heart_rate,
            }
