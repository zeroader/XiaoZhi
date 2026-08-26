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
from dataclasses import dataclass
from typing import Any, Deque, List, Optional


@dataclass
class FrameRecord:
    """缓存的一帧"""
    frame_id: int
    timestamp: float          # 接收时刻（unix 秒），用于计算采样率
    image: Any                # RGB numpy 数组（已降采样，节省内存）
    width: int = 0            # 原始宽度
    height: int = 0           # 原始高度


@dataclass
class RppgSample:
    """A compact forehead-color sample for long rPPG measurement windows."""
    timestamp: float
    rgb: Optional[Any]
    face_confidence: float = 0.0


class ServerState:
    """全局状态：最近帧缓存 + 最新各任务结果"""

    def __init__(self, buffer_maxlen: int = 30, rppg_maxlen: int = 1800):
        self.buffer_maxlen = buffer_maxlen
        self.rppg_maxlen = rppg_maxlen
        self._lock = threading.RLock()
        self.frame_buffer: Deque[FrameRecord] = deque(maxlen=buffer_maxlen)
        self.rppg_samples: Deque[RppgSample] = deque(maxlen=rppg_maxlen)
        self.latest_face: Optional[dict] = None
        self.latest_emotion: Optional[dict] = None
        self.latest_posture: Optional[dict] = None
        self.latest_heart_rate: Optional[dict] = None
        self.latest_blood_pressure: Optional[dict] = None
        self._frame_seq = 0

    # ---------- 帧缓存 ----------

    def push_frame(self, image, width: int = 0, height: int = 0,
                   timestamp: Optional[float] = None) -> int:
        """保存当前帧到缓存，返回自增 frame_id。"""
        with self._lock:
            self._frame_seq += 1
            rec = FrameRecord(
                frame_id=self._frame_seq,
                timestamp=time.time() if timestamp is None else float(timestamp),
                image=image,
                width=width,
                height=height,
            )
            self.frame_buffer.append(rec)
            return rec.frame_id

    def push_rppg_sample(self, timestamp: float, rgb,
                         face_confidence: float = 0.0) -> bool:
        """Record a compact rPPG sample and discard duplicate/out-of-order timestamps."""
        with self._lock:
            if self.rppg_samples and timestamp <= self.rppg_samples[-1].timestamp:
                return False
            self.rppg_samples.append(RppgSample(
                timestamp=float(timestamp), rgb=rgb,
                face_confidence=float(face_confidence)))
            return True

    def get_recent_rppg_samples(self, seconds: Optional[float] = None) -> List[RppgSample]:
        """Return compact rPPG samples in chronological order."""
        with self._lock:
            samples = list(self.rppg_samples)
        if not samples or seconds is None or seconds <= 0:
            return samples
        cutoff = samples[-1].timestamp - float(seconds)
        return [sample for sample in samples if sample.timestamp >= cutoff]

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

    def set_blood_pressure(self, result: dict):
        with self._lock:
            self.latest_blood_pressure = result

    def snapshot(self) -> dict:
        """汇总最新结果，供 /state 查询。"""
        with self._lock:
            return {
                "buffer_frames": len(self.frame_buffer),
                "buffer_maxlen": self.buffer_maxlen,
                "rppg_samples": len(self.rppg_samples),
                "rppg_maxlen": self.rppg_maxlen,
                "latest_face": self.latest_face,
                "latest_emotion": self.latest_emotion,
                "latest_posture": self.latest_posture,
                "latest_heart_rate": self.latest_heart_rate,
                "latest_blood_pressure": self.latest_blood_pressure,
            }
