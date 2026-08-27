"""Experimental camera-based blood-pressure inference.

The component performs only these production-safe stages:
  frame -> face forehead RGB -> full rPPG/BVP waveform -> quality gate -> BP model.

An estimate is returned only when a trained ONNX model is explicitly configured. The
model must be exported from a subject-independent rPPG-BP training run; this module
does not synthesize SBP/DBP from heart rate or hand-written rules.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from detectors.heart_detector import HeartRateDetector, METHODS, extract_rppg_waveform


EXPERIMENTAL_DISCLAIMER = (
    "Experimental, non-medical measurement. Do not use for diagnosis, treatment, or "
    "emergency decisions."
)


class OnnxBloodPressureModel:
    """Small deployment adapter for an rPPG-BP model exported to ONNX."""

    def __init__(self, model_path: Optional[str], config_path: Optional[str] = None):
        self.path = Path(model_path) if model_path else None
        self.config = {}
        self.session = None
        self.input_name = None
        self.input_rank = None
        self.input_shape = None
        self.input_length = None
        self.error = None

        if config_path:
            try:
                with Path(config_path).open("r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                self.error = f"invalid_bp_model_config: {exc}"
                return
        if not self.path:
            self.error = "bp_model_not_loaded"
            return
        if not self.path.is_file():
            self.error = f"bp_model_missing: {self.path}"
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(self.path), providers=["CPUExecutionProvider"])
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            self.input_rank = len(model_input.shape)
            self.input_shape = model_input.shape
            if self.input_rank not in (2, 3):
                raise ValueError("model input must be [B,T], [B,1,T], or [B,T,1]")
            if self.input_rank == 3:
                dimensions = model_input.shape[1:]
                if dimensions[0] == 1:
                    static_length = dimensions[1]
                elif dimensions[1] == 1:
                    static_length = dimensions[0]
                else:
                    raise ValueError("rank-3 model input must have one channel")
            else:
                static_length = model_input.shape[-1]
            if isinstance(static_length, int) and static_length > 0:
                self.input_length = static_length
            self.input_length = int(self.config.get("input_length", self.input_length or 0)) or None
        except Exception as exc:
            self.session = None
            self.error = f"bp_model_load_failed: {exc}"

    @property
    def ready(self) -> bool:
        return self.session is not None

    def describe(self) -> dict:
        return {
            "ready": self.ready,
            "path": str(self.path) if self.path else None,
            "input_shape": self.input_shape,
            "input_length": self.input_length,
            "error": self.error,
        }

    def predict(self, waveform: np.ndarray) -> tuple:
        if not self.ready:
            raise RuntimeError(self.error or "bp_model_not_loaded")
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self.input_length:
            source_x = np.linspace(0.0, 1.0, len(signal), dtype=np.float32)
            target_x = np.linspace(0.0, 1.0, self.input_length, dtype=np.float32)
            signal = np.interp(target_x, source_x, signal).astype(np.float32)
        mean = float(self.config.get("input_mean", 0.0))
        std = float(self.config.get("input_std", 1.0))
        signal = (signal - mean) / max(std, 1e-9)
        if self.input_rank == 3:
            dimensions = self.input_shape[1:]
            if dimensions[0] == 1:
                model_input = signal.reshape(1, 1, -1)
            else:
                # Zenodo's Keras ResNet uses [B,T,1].
                model_input = signal.reshape(1, -1, 1)
        else:
            model_input = signal.reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: model_input})
        if len(outputs) >= 2:
            # Some Keras models expose SBP and DBP as two output tensors.
            sbp = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            dbp = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
            if not len(sbp) or not len(dbp):
                raise ValueError("bp_model_output_must_contain_sbp_dbp")
            values = np.asarray([sbp[0], dbp[0]], dtype=np.float32)
        else:
            values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if len(values) < 2:
                raise ValueError("bp_model_output_must_contain_sbp_dbp")
            values = values[:2]
        if not np.all(np.isfinite(values)):
            raise ValueError("bp_model_output_must_contain_sbp_dbp")
        return float(values[0]), float(values[1])


class BloodPressureDetector:
    """Collects forehead samples and runs an optional trained BP network."""

    def __init__(self, face_detector, method: str = "pos", model_path: Optional[str] = None,
                 config_path: Optional[str] = None, window_seconds: float = 30.0,
                 min_fps: float = 10.0, min_face_ratio: float = 0.80,
                 min_snr: float = 2.0, relaxed_quality: bool = True):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.face_detector = face_detector
        self.method = method
        self.window_seconds = float(window_seconds)
        self.min_fps = float(min_fps)
        self.min_face_ratio = float(min_face_ratio)
        self.min_snr = float(min_snr)
        self.relaxed_quality = bool(relaxed_quality)
        self._roi = HeartRateDetector(None, method=method)
        self.model = OnnxBloodPressureModel(model_path, config_path)
        self._last_face = None

    def sample_frame(self, image_rgb: np.ndarray) -> tuple:
        """Return (forehead_rgb, confidence); rgb is None when no usable face exists."""
        self._last_face = None
        if self.face_detector is None:
            return None, 0.0
        h, w = image_rgb.shape[:2]
        face = self.face_detector.detect_primary_face(image_rgb, w, h)
        self._last_face = face
        if face is None:
            return None, 0.0
        rgb = self._roi._forehead_mean_rgb(image_rgb, face["bbox"])
        return rgb, float(face["confidence"])

    def get_last_face(self) -> Optional[dict]:
        """Return the face detected while sampling the most recent frame."""
        return self._last_face

    def _quality(self, samples: Iterable) -> tuple:
        samples = list(samples)
        if len(samples) < 2:
            return None, {"accepted": False, "reason": "collecting", "duration_s": 0.0,
                          "samples": len(samples), "valid_face_ratio": 0.0}
        duration = float(samples[-1].timestamp - samples[0].timestamp)
        valid = [sample for sample in samples if sample.rgb is not None]
        ratio = len(valid) / len(samples)
        details = {
            "duration_s": round(duration, 2),
            "samples": len(samples),
            "valid_samples": len(valid),
            "valid_face_ratio": round(ratio, 3),
            "required_window_s": self.window_seconds,
            "required_min_fps": self.min_fps,
        }
        if duration < self.window_seconds * 0.95:
            details.update(accepted=False, reason="collecting")
            return None, details
        # Prototype mode prioritizes exercising the full server -> model path.
        # It still needs enough valid samples to construct a waveform.
        if len(valid) < 8:
            details.update(accepted=False, reason="insufficient_valid_face_samples")
            return None, details
        if not self.relaxed_quality and ratio < self.min_face_ratio:
            details.update(accepted=False, reason="face_visibility_too_low")
            return None, details
        if not self.relaxed_quality and len(valid) < int(
                self.window_seconds * self.min_fps * self.min_face_ratio):
            details.update(accepted=False, reason="sampling_rate_too_low")
            return None, details
        rgb = np.stack([sample.rgb for sample in valid])
        times = np.asarray([sample.timestamp for sample in valid], dtype=np.float64)
        try:
            waveform = extract_rppg_waveform(rgb, times, method=self.method)
        except ValueError as exc:
            details.update(accepted=False, reason=str(exc))
            return None, details
        details.update(
            fs=round(waveform["fs"], 2),
            waveform_frames=waveform["frames_used"],
            snr=round(waveform["snr"], 3),
        )
        if not self.relaxed_quality and waveform["fs"] < self.min_fps:
            details.update(accepted=False, reason="sampling_rate_too_low")
            return None, details
        if not self.relaxed_quality and waveform["snr"] < self.min_snr:
            details.update(accepted=False, reason="rppg_signal_too_noisy")
            return None, details
        details.update(accepted=True, reason="relaxed_quality" if self.relaxed_quality else "ok")
        return waveform, details

    def detect(self, samples: Iterable) -> dict:
        waveform, quality = self._quality(samples)
        base = {
            "experimental_only": True,
            "disclaimer": EXPERIMENTAL_DISCLAIMER,
            "method": self.method,
            "quality": quality,
        }
        if waveform is None:
            return {"status": "collecting" if quality["reason"] == "collecting" else "rejected", **base}
        if not self.model.ready:
            return {"status": "waveform_ready", "error": self.model.error,
                    "model": self.model.describe(), **base}
        try:
            sbp, dbp = self.model.predict(waveform["waveform"])
            if not (50.0 <= sbp <= 250.0 and 30.0 <= dbp <= 150.0 and sbp > dbp):
                raise ValueError("bp_model_output_out_of_range")
        except (RuntimeError, ValueError) as exc:
            return {"status": "model_error", "error": str(exc),
                    "model": self.model.describe(), **base}
        return {
            "status": "ready",
            "sbp_mmHg": round(sbp, 1),
            "dbp_mmHg": round(dbp, 1),
            "model": self.model.describe(),
            **base,
        }
