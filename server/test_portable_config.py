import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import main
import runtime_config
import lamp_controller
import numpy as np
from detectors.face_detector import FaceDetector


class PortableConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "server_config.json"

    def write(self, settings):
        self.config.write_text(json.dumps(settings), encoding="utf-8-sig")

    def test_external_settings_and_cli_override(self):
        self.write({"port": 19001, "buffer_size": 22, "bp_strict_quality": True,
                    "face_model": "models/face.onnx", "lamp_config": "config/lamp.json"})
        args = main.parse_args(["--config", str(self.config), "--port", "19002",
                                "--no-bp-strict-quality"])
        self.assertEqual(args.port, 19002)
        self.assertEqual(args.buffer_size, 22)
        self.assertFalse(args.bp_strict_quality)
        self.assertEqual(Path(args.face_model), self.root / "models" / "face.onnx")
        self.assertEqual(Path(args.lamp_config), self.root / "config" / "lamp.json")

    def test_bad_settings_and_missing_explicit_config_fail(self):
        for data in ({"port": 65536}, {"port": True}, {"bp_strict_quality": "false"},
                     {"buffer_size": 0}, {"heart_method": "typo"}, {"unknown": 123}, []):
            self.write(data)
            with self.subTest(data=data), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main.parse_args(["--config", str(self.config)])
                self.assertEqual(raised.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--config", str(self.root / "missing.json")])

    def test_frozen_app_uses_executable_directory(self):
        spec = importlib.util.spec_from_file_location("frozen_config", runtime_config.__file__)
        frozen = importlib.util.module_from_spec(spec)
        with patch.object(sys, "frozen", True, create=True), patch.object(
                sys, "executable", str(self.root / "XiaozhiServer.exe")):
            spec.loader.exec_module(frozen)
        self.assertEqual(frozen.APP_DIR, self.root)
        self.assertEqual(frozen.DEFAULT_MODELS_DIR, self.root / "models")

    def test_source_defaults_do_not_depend_on_working_directory(self):
        args = main.parse_args([])
        self.assertTrue(Path(args.face_model).is_absolute())
        self.assertEqual(Path(args.bp_model).parent, runtime_config.DEFAULT_MODELS_DIR)

    def test_lamp_config_can_be_external_and_have_utf8_bom(self):
        lamp_path = self.root / "lamp.json"
        lamp_path.write_text(json.dumps({"token": "test-token", "ip": "auto"}), encoding="utf-8-sig")
        self.assertEqual(lamp_controller._load_config(lamp_path)["token"], "test-token")

    def test_face_model_in_chinese_directory(self):
        model = self.root / "中文 空格" / "人脸.onnx"
        model.parent.mkdir()
        model.write_bytes((runtime_config.DEFAULT_MODELS_DIR / "face_detection_yunet_2023mar.onnx").read_bytes())
        detector = FaceDetector(str(model))
        self.assertEqual(detector.detect(np.zeros((240, 320, 3), dtype=np.uint8), 320, 240), [])


if __name__ == "__main__":
    unittest.main()
