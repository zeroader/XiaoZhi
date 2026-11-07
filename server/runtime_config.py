"""Editable settings live next to the executable, never in its temp directory."""

import json
import math
import sys
from pathlib import Path


FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = APP_DIR / "models" if FROZEN else APP_DIR.parent / "models"

DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8291,
    "face_model": str(DEFAULT_MODELS_DIR / "face_detection_yunet_2023mar.onnx"),
    "emotion_model": str(DEFAULT_MODELS_DIR / "enet_b0_8_best_afew.onnx"),
    "pose_model": str(DEFAULT_MODELS_DIR / "yolov8n_pose.onnx"),
    "buffer_size": 60,
    "heart_method": "pos",
    "bp_method": "pos",
    "bp_model": str(DEFAULT_MODELS_DIR / "resnet_ppg_nonmixed.onnx"),
    "bp_model_config": str(DEFAULT_MODELS_DIR / "resnet_ppg_nonmixed.json"),
    "bp_window_seconds": 7.0,
    "bp_strict_quality": False,
    "rppg_max_samples": 1800,
    "lamp_config": str(APP_DIR / "lamp_config.json"),
}
PATH_KEYS = ("face_model", "emotion_model", "pose_model", "bp_model", "bp_model_config", "lamp_config")


def validate_settings(settings):
    """Catch malformed JSON settings before allocating buffers/loading models."""
    for key, value in settings.items():
        if key not in DEFAULTS:
            raise ValueError(f"Unknown setting: {key}")
        default = DEFAULTS[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = type(value) is int and value > 0
        elif isinstance(default, float):
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid value for {key}: {value!r}")
        if key == "port" and value > 65535:
            raise ValueError("port must be between 1 and 65535")
        if key in ("heart_method", "bp_method") and value not in ("green", "pos", "chrom"):
            raise ValueError(f"{key} must be green, pos or chrom")
        if key in ("host", "face_model", "lamp_config") and not value.strip():
            raise ValueError(f"{key} must not be empty")


def load_settings(config_path, required=False):
    path = Path(config_path).resolve()
    if not path.is_file():
        if required:
            raise ValueError(f"Config file not found: {path}")
        return {}
    try:
        with path.open(encoding="utf-8-sig") as stream:
            settings = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    validate_settings(settings)
    return settings


def resolve_settings_paths(settings, config_path):
    """Relative paths (including CLI overrides) use the config file's directory."""
    root = Path(config_path).resolve().parent
    for key in PATH_KEYS:
        value = settings[key]
        if value:
            path = Path(value).expanduser()
            settings[key] = str((root / path).resolve() if not path.is_absolute() else path)
    return settings
