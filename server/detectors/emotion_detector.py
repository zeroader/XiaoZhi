"""
情绪识别器 - EmotiEffLib enet_b0_8_best_afew (AffectNet 8 分类)

模型: models/enet_b0_8_best_afew.onnx
来源: https://github.com/sb-ai-lab/EmotiEffLib
推理: onnxruntime（参考 av-savchenko/hsemotion-onnx 的 facial_emotions.py 用法）

预处理（与 hsemotion 一致）:
  resize -> 224x224, /255, 按通道 (mean, std) 归一化, HWC -> NCHW

输出: 8 分类 logits -> softmax
  Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
"""

import cv2
import numpy as np
import onnxruntime as ort

# AffectNet 8 分类标签（小写，便于协议一致）
EMOTION_LABELS_AFFECTNET = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise",
]
# FER2013 7 分类标签（当模型输出 7 类时使用）
EMOTION_LABELS_FER2013 = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
]

INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)   # ImageNet 均值
STD = (0.229, 0.224, 0.225)    # ImageNet 标准差


class EmotionDetector:
    """情绪分类器（onnxruntime）"""

    def __init__(self, model_path: str):
        self.ort_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        out_shape = self.ort_session.get_outputs()[0].shape
        n_out = int(out_shape[-1]) if out_shape and out_shape[-1] is not None else 8

        # 按输出类别数选择标签集（8 -> AffectNet, 7 -> FER2013）
        if n_out == 8:
            self.labels = list(EMOTION_LABELS_AFFECTNET)
        elif n_out == 7:
            self.labels = list(EMOTION_LABELS_FER2013)
        else:
            self.labels = [f"class_{i}" for i in range(n_out)]
        print(f"[Model] Emotion loaded: {model_path} "
              f"(onnxruntime, input={INPUT_SIZE}x{INPUT_SIZE}, classes={len(self.labels)})")

    # ---------- 预处理 ----------

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE)) / 255.0
        for i in range(3):
            x[..., i] = (x[..., i] - MEAN[i]) / STD[i]
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]

    # ---------- 推理 ----------

    def detect(self, face_rgb: np.ndarray) -> dict:
        """对裁剪后的人脸区域做情绪分类。

        返回: {"label": str, "confidence": float}
        """
        if face_rgb is None or face_rgb.size == 0:
            return None

        blob = self._preprocess(face_rgb)
        logits = self.ort_session.run(None, {self.input_name: blob})[0][0]

        # softmax -> 概率
        e = np.exp(logits - np.max(logits))
        probs = e / (e.sum() + 1e-9)
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return {"label": label, "confidence": round(float(probs[idx]), 4)}
