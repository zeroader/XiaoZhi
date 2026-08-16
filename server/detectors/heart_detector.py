"""
心率检测器 - rPPG（多方法可切换）

支持三种无监督 rPPG 信号提取方法（借鉴 ubicomplab/rPPG-Toolbox）:
  - green: GREEN 基础法（Verkruysse 2008），仅用 G 通道均值
  - pos:   POS（Wang 2017），投影到正交皮肤色空间，运动鲁棒性最好
  - chrom: CHROM（de Haan 2013），色度归一化，对光照变化鲁棒

统一后处理: 平滑先验 detrend -> 带通滤波 (0.7~4Hz) -> FFT -> 峰值频率 * 60 = BPM

rPPG 不做定时采集：由 ESP32 每 10s 触发一次 heart_rate 任务，
服务器直接读取 frame_buffer 中已缓存的最近帧。
"""

import math

import numpy as np
from scipy import signal as sp_signal
from scipy import sparse

# 信号处理参数
BANDPASS_LOW = 0.7    # Hz (42 BPM)
BANDPASS_HIGH = 4.0   # Hz (240 BPM)
FFT_PADDING = 4       # 频域插值倍数，提升频率分辨率
MIN_FRAMES = 8        # 最少需要的有效帧数

# 支持的方法
METHODS = ("green", "pos", "chrom")

# 前额 ROI（相对人脸 bbox 比例）
ROI_X = 0.35
ROI_Y = 0.10
ROI_W = 0.30
ROI_H = 0.25


def detrend(input_signal, lambda_value: float = 100.0) -> np.ndarray:
    """平滑先验去趋势（Tarvainen 2002），移植自 rPPG-Toolbox utils.py。

    通过正则化二阶差分矩阵消除缓慢漂移，优于简单线性去趋势。
    """
    sig = np.asarray(input_signal, dtype=float).reshape(-1)
    n = len(sig)
    if n < 4:
        return sig - sig.mean()

    I = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    D = sparse.spdiags(np.array([ones, minus_twos, ones]),
                       np.array([0, 1, 2]), n - 2, n).toarray()
    inv = np.linalg.inv(I + (lambda_value ** 2) * (D.T @ D))
    return (I - inv) @ sig


class HeartRateDetector:
    """rPPG 心率检测器（多方法可切换）"""

    def __init__(self, face_detector, method: str = "pos",
                 band_low: float = BANDPASS_LOW,
                 band_high: float = BANDPASS_HIGH,
                 min_frames: int = MIN_FRAMES):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.face_detector = face_detector
        self.method = method
        self.band_low = band_low
        self.band_high = band_high
        self.min_frames = min_frames

    # ---------- 信号提取 ----------

    def _forehead_mean_rgb(self, image_rgb: np.ndarray, bbox: dict) -> np.ndarray:
        """裁剪前额 ROI 并返回 RGB 三通道均值。"""
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
        return roi.reshape(-1, 3).mean(axis=0)   # [R, G, B]

    def _extract_signal(self, frames) -> tuple:
        """从缓存帧提取 RGB 均值信号与时间戳。

        返回: (rgb (N,3), times, n_used)  无足够人脸时返回 (None, None, 0)
        """
        rgb_list = []
        times = []
        for rec in frames:
            bbox = None
            if self.face_detector is not None:
                face = self.face_detector.detect_primary_face(
                    rec.image, rec.width, rec.height)
                bbox = face["bbox"] if face else None
            if bbox is None:
                continue
            rgb = self._forehead_mean_rgb(rec.image, bbox)
            if rgb is None:
                continue
            rgb_list.append(rgb)
            times.append(rec.timestamp)

        if len(rgb_list) < self.min_frames:
            return None, None, len(rgb_list)
        return np.asarray(rgb_list, dtype=np.float64), \
            np.asarray(times, dtype=np.float64), len(rgb_list)

    @staticmethod
    def _resample_uniform(rgb: np.ndarray, times: np.ndarray) -> np.ndarray:
        """将非等间隔采样信号重采样到均匀时间轴（POS/CHROM 假设等间隔采样）。"""
        n = len(rgb)
        if n < 3:
            return rgb
        t = times - times[0]
        if t[-1] <= 0:
            return rgb
        t_uniform = np.linspace(0, t[-1], n)
        out = np.empty_like(rgb)
        for c in range(3):
            out[:, c] = np.interp(t_uniform, t, rgb[:, c])
        return out

    # ---------- 信号生成（按方法） ----------

    def _generate_bvp(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        """根据 self.method 生成 BVP 信号。"""
        if self.method == "pos":
            return self._pos(rgb, fs)
        if self.method == "chrom":
            return self._chrom(rgb, fs)
        return rgb[:, 1]   # green: G 通道

    def _pos(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        """POS 算法（Wang 2017），移植自 rPPG-Toolbox POS_WANG.py。

        1.6s 滑动窗口内: 逐样本除以其窗口均值 -> 投影到正交色空间 ->
        用标准差的比值加权组合 -> 累加得到 BVP。
        """
        win_sec = 1.6
        n = rgb.shape[0]
        h = np.zeros(n)
        l = max(1, math.ceil(win_sec * fs))
        for i in range(n):
            m = i - l
            if m >= 0:
                cn = rgb[m:i, :] / np.mean(rgb[m:i, :], axis=0)     # (l,3)
                s = np.array([[0, 1, -1], [-2, 1, 1]]) @ cn.T       # (2,l)
                std0 = np.std(s[0, :]); std1 = np.std(s[1, :])
                if std1 < 1e-9:
                    continue
                h_win = s[0, :] + (std0 / std1) * s[1, :]
                h[m:i] += h_win - np.mean(h_win)
        return h

    def _chrom(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        """CHROM 算法（de Haan 2013），移植自 rPPG-Toolbox CHROME_DEHAAN.py。

        色度信号 Xs/Ys -> 带通滤波 -> 用 std 比值消除色度分量 -> 分段拼接。
        """
        lpf = 0.7
        hpf = 2.5
        win_sec = 1.6
        fn = rgb.shape[0]
        nyq = fs / 2.0
        if nyq <= hpf:
            return rgb[:, 1]
        b, a = sp_signal.butter(3, [max(lpf / nyq, 1e-4), min(hpf / nyq, 1 - 1e-4)],
                                "bandpass")
        win_l = math.ceil(win_sec * fs)
        if win_l % 2:
            win_l += 1
        n_win = math.floor((fn - win_l // 2) / (win_l // 2))
        if n_win < 1:
            return rgb[:, 1]
        total_len = (win_l // 2) * (n_win + 1)
        s = np.zeros(total_len)

        win_s = 0
        win_m = win_l // 2
        win_e = win_l
        for _ in range(n_win):
            rgb_base = np.mean(rgb[win_s:win_e, :], axis=0)
            rgb_norm = rgb[win_s:win_e, :] / rgb_base
            xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
            ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
            xf = sp_signal.filtfilt(b, a, xs, axis=0)
            yf = sp_signal.filtfilt(b, a, ys, axis=0)
            std_y = np.std(yf)
            alpha = np.std(xf) / std_y if std_y > 1e-9 else 0.0
            s_win = (xf - alpha * yf) * sp_signal.windows.hann(win_l)
            s[win_s:win_m] += s_win[:win_l // 2]
            s[win_m:win_e] = s_win[win_l // 2:]
            win_s = win_m
            win_m = win_s + win_l // 2
            win_e = win_s + win_l
        return s

    # ---------- 信号处理 ----------

    def _bandpass_fft(self, sig: np.ndarray, fs: float) -> np.ndarray:
        """FFT 域带通滤波（回退方案）。"""
        n = len(sig)
        if n < 4 or fs <= 0:
            return sig
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        fft = np.fft.rfft(sig)
        mask = (freqs >= self.band_low) & (freqs <= self.band_high)
        fft[~mask] = 0.0
        return np.fft.irfft(fft, n)

    def _bandpass(self, sig: np.ndarray, fs: float) -> np.ndarray:
        """带通滤波：优先 scipy butter+filtfilt，短信号回退 FFT 掩码。"""
        nyq = fs / 2.0
        if nyq <= self.band_high:
            return self._bandpass_fft(sig, fs)
        try:
            low = max(self.band_low / nyq, 1e-4)
            high = min(self.band_high / nyq, 1 - 1e-4)
            b, a = sp_signal.butter(1, [low, high], btype="bandpass")
            padlen = 3 * max(len(a), len(b))
            if len(sig) <= padlen:
                return self._bandpass_fft(sig, fs)
            return sp_signal.filtfilt(b, a, sig)
        except Exception:
            return self._bandpass_fft(sig, fs)

    def _estimate_bpm(self, sig: np.ndarray, fs: float) -> tuple:
        """detrend + 带通 + FFT，估计 BPM 与置信度。

        返回: (bpm, confidence)  无有效峰值时返回 (None, 0)
        """
        sig = detrend(sig)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        sig = self._bandpass(sig, fs)

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
        return round(float(bpm), 1), round(float(conf), 4)

    # ---------- 对外接口 ----------

    def detect(self, frames) -> dict:
        """HeartRateDetector.detect(frames)

        参数: frames - ServerState.get_recent_frames() 的结果
        返回: {"bpm": 76, "confidence": 0.85} 或 {"error": "..."}
        """
        rgb, times, n_used = self._extract_signal(frames)
        if rgb is None:
            return {"error": f"insufficient_frames ({n_used}/{self.min_frames} valid)"}

        # 采样率: 由帧时间戳估计（帧间中位数间隔）
        dts = np.diff(times)
        dts = dts[dts > 0]
        if dts.size == 0:
            return {"error": "invalid_timestamps"}
        fs = 1.0 / float(np.median(dts))

        rgb = self._resample_uniform(rgb, times)
        bvp = self._generate_bvp(rgb, fs)
        bpm, conf = self._estimate_bpm(bvp, fs)
        if bpm is None:
            return {"error": "no_peak_in_band"}

        return {"bpm": round(float(bpm), 1), "confidence": round(float(conf), 4),
                "method": self.method,
                "fs": round(float(fs), 2), "frames_used": int(n_used)}
