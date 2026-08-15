"""
心率检测器 - rPPG (传统算法，第一版)

原理: 从缓存视频帧中提取前额 ROI 的 RGB 均值（使用 G 通道），
     得到 PPG 信号 -> 带通滤波 (0.7~4Hz) -> FFT -> 峰值频率 * 60 = BPM

rPPG 不做定时采集：由 ESP32 每 10s 触发一次 heart_rate 任务，
服务器直接读取 frame_buffer 中已缓存的最近帧。
"""

import numpy as np

# 信号处理参数
BANDPASS_LOW = 0.7    # Hz (42 BPM)
BANDPASS_HIGH = 4.0   # Hz (240 BPM)
FFT_PADDING = 4       # 频域插值倍数，提升频率分辨率
MIN_FRAMES = 8        # 最少需要的有效帧数

# 前额 ROI（相对人脸 bbox 比例）
ROI_X = 0.35
ROI_Y = 0.10
ROI_W = 0.30
ROI_H = 0.25


class HeartRateDetector:
    """rPPG 心率检测器"""

    def __init__(self, face_detector, band_low: float = BANDPASS_LOW,
                 band_high: float = BANDPASS_HIGH, min_frames: int = MIN_FRAMES):
        self.face_detector = face_detector
        self.band_low = band_low
        self.band_high = band_high
        self.min_frames = min_frames

    # ---------- 信号提取 ----------

    def _forehead_mean_green(self, image_rgb: np.ndarray, bbox: dict) -> float:
        """裁剪前额 ROI 并返回 G 通道均值。"""
        h, w = image_rgb.shape[:2]
        x = int(bbox["x"] + bbox["width"] * ROI_X)
        y = int(bbox["y"] + bbox["height"] * ROI_Y)
        rw = max(1, int(bbox["width"] * ROI_W))
        rh = max(1, int(bbox["height"] * ROI_H))
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi = image_rgb[y:y + rh, x:x + rw]
        if roi.size == 0:
            return None
        return float(roi[:, :, 1].mean())   # G 通道

    def _extract_signal(self, frames) -> tuple:
        """从缓存帧提取 PPG 信号（G 通道）与时间戳。

        返回: (signal, times, n_used)  无足够人脸时返回 (None, None, 0)
        """
        signal = []
        times = []
        for rec in frames:
            bbox = None
            if self.face_detector is not None:
                face = self.face_detector.detect_primary_face(
                    rec.image, rec.width, rec.height)
                bbox = face["bbox"] if face else None
            if bbox is None:
                continue
            g = self._forehead_mean_green(rec.image, bbox)
            if g is None:
                continue
            signal.append(g)
            times.append(rec.timestamp)

        if len(signal) < self.min_frames:
            return None, None, len(signal)
        return np.array(signal, dtype=np.float64), np.array(times, dtype=np.float64), len(signal)

    # ---------- 信号处理 ----------

    @staticmethod
    def _detrend(x: np.ndarray) -> np.ndarray:
        """去趋势（减去线性拟合）。"""
        n = len(x)
        t = np.arange(n)
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef

    @staticmethod
    def _bandpass_fft(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
        """FFT 域带通滤波。"""
        n = len(sig)
        if n < 4 or fs <= 0:
            return sig
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        fft = np.fft.rfft(sig)
        mask = (freqs >= low) & (freqs <= high)
        fft[~mask] = 0.0
        return np.fft.irfft(fft, n)

    def _estimate_bpm(self, sig: np.ndarray, fs: float) -> tuple:
        """带通滤波 + FFT，估计 BPM 与置信度。

        返回: (bpm, confidence)  无有效峰值时返回 (None, 0)
        """
        sig = self._detrend(sig)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        sig = self._bandpass_fft(sig, fs, self.band_low, self.band_high)

        n = len(sig)
        nfft = n * FFT_PADDING
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        spectrum = np.abs(np.fft.rfft(sig, nfft))

        in_band = (freqs >= self.band_low) & (freqs <= self.band_high)
        if not in_band.any():
            return None, 0.0

        band_spectrum = spectrum.copy()
        band_spectrum[~in_band] = 0.0
        peak_idx = int(np.argmax(band_spectrum))
        peak_freq = freqs[peak_idx]
        bpm = peak_freq * 60.0

        if bpm < 30 or bpm > 240:
            return None, 0.0

        # 置信度: 峰值功率占带内总功率比例
        total_power = np.sum(band_spectrum) + 1e-9
        conf = float(spectrum[peak_idx] / total_power)
        conf = max(0.0, min(1.0, conf))
        return round(bpm, 1), round(conf, 4)

    # ---------- 对外接口 ----------

    def detect(self, frames) -> dict:
        """HeartRateDetector.detect(frames)

        参数: frames - ServerState.get_recent_frames() 的结果
        返回: {"bpm": 76, "confidence": 0.85} 或 {"error": "..."}
        """
        signal, times, n_used = self._extract_signal(frames)
        if signal is None:
            return {"error": f"insufficient_frames ({n_used}/{self.min_frames} valid)"}

        # 采样率: 由帧时间戳估计（帧间中位数间隔）
        dts = np.diff(times)
        dts = dts[dts > 0]
        if dts.size == 0:
            return {"error": "invalid_timestamps"}
        fs = 1.0 / float(np.median(dts))

        bpm, conf = self._estimate_bpm(signal, fs)
        if bpm is None:
            return {"error": "no_peak_in_band"}

        return {"bpm": round(float(bpm), 1), "confidence": round(float(conf), 4),
                "fs": round(float(fs), 2), "frames_used": int(n_used)}
