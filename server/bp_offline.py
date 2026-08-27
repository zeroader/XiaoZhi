"""Offline MP4 validation for the experimental rPPG -> blood-pressure pipeline.

Run this before connecting an ESP32. It saves the extracted full waveform so it can
be compared with synchronized ground-truth PPG, then optionally runs an exported
rPPG-BP ONNX model.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import state
from detectors.blood_pressure_detector import BloodPressureDetector
from detectors.face_detector import FaceDetector
from detectors.heart_detector import METHODS, extract_rppg_waveform


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FACE_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def main():
    parser = argparse.ArgumentParser(description="Offline MP4 rPPG / experimental BP validation")
    parser.add_argument("video", help="input MP4 path")
    parser.add_argument("--face-model", default=str(DEFAULT_FACE_MODEL), help="YuNet ONNX path")
    parser.add_argument("--method", choices=METHODS, default="pos", help="rPPG extraction method")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="last N seconds to evaluate")
    parser.add_argument("--bp-model", default="", help="trained rPPG-BP ONNX model (optional)")
    parser.add_argument("--bp-model-config", default="", help="model JSON config (optional)")
    parser.add_argument("--output", default="", help="waveform .npz output path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")
    face_path = Path(args.face_model)
    if not face_path.is_file():
        raise SystemExit(f"face model not found: {face_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    detector = BloodPressureDetector(
        FaceDetector(str(face_path)), method=args.method, model_path=args.bp_model or None,
        config_path=args.bp_model_config or None, window_seconds=args.window_seconds)

    samples = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else index / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_rgb, confidence = detector.sample_frame(rgb)
        samples.append(state.RppgSample(timestamp, roi_rgb, confidence))
        index += 1
    capture.release()

    if samples:
        cutoff = samples[-1].timestamp - args.window_seconds
        samples = [sample for sample in samples if sample.timestamp >= cutoff]
    result = detector.detect(samples)

    valid = [sample for sample in samples if sample.rgb is not None]
    output_path = Path(args.output) if args.output else video_path.with_suffix(".rppg.npz")
    if len(valid) >= 8:
        waveform = extract_rppg_waveform(
            np.stack([sample.rgb for sample in valid]),
            np.asarray([sample.timestamp for sample in valid]), method=args.method)
        np.savez_compressed(
            output_path,
            waveform=waveform["waveform"],
            timestamps=np.asarray([sample.timestamp for sample in valid]),
            fs=np.asarray([waveform["fs"]]),
            snr=np.asarray([waveform["snr"]]),
        )
        result["waveform_file"] = str(output_path)
    else:
        result["waveform_file"] = None

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
