"""Convert the pretrained Zenodo ResNet PPG blood-pressure model to ONNX.

The source model expects one 7-second signal window containing 875 samples
(125 Hz) and predicts systolic/diastolic blood pressure.  The generated ONNX
file is compatible with detectors.blood_pressure_detector.OnnxBloodPressureModel.

Install the export-only dependencies first, for example:
    pip install tensorflow tf2onnx onnx onnxruntime

Usage:
    python server/convert_resnet_bp.py \
        --h5 C:\\models\\resnet_ppg_nonmixed.h5 \
        --output C:\\models\\resnet_ppg_nonmixed.onnx
"""

import argparse
import json
from pathlib import Path


def _load_model(path: Path):
    import tensorflow as tf

    custom_objects = {"ReLU": tf.keras.layers.ReLU}
    try:
        from kapre import STFT, Magnitude, MagnitudeToDecibel

        custom_objects.update({
            "STFT": STFT,
            "Magnitude": Magnitude,
            "MagnitudeToDecibel": MagnitudeToDecibel,
        })
    except ImportError:
        # ResNet normally does not need kapre, but older checkpoints may carry
        # a serialized custom-object reference.
        pass

    return tf.keras.models.load_model(
        str(path), custom_objects=custom_objects, compile=False)


def _shape_for_signature(shape):
    shape = list(shape)
    if not shape:
        raise ValueError("model has no input shape")
    shape[0] = None
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Zenodo .h5 model")
    parser.add_argument("--output", required=True, help="output .onnx path")
    parser.add_argument("--config", default="", help="output JSON config path")
    parser.add_argument("--input-mean", type=float, default=0.0)
    parser.add_argument("--input-std", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    h5_path = Path(args.h5)
    output_path = Path(args.output)
    if not h5_path.is_file():
        raise SystemExit(f"model not found: {h5_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    import tf2onnx

    model = _load_model(h5_path)
    if len(model.inputs) != 1:
        raise ValueError(
            f"expected one model input, got {len(model.inputs)}; "
            "this adapter accepts only the signal input")

    input_shape = _shape_for_signature(model.inputs[0].shape)
    model_dimensions = list(model.inputs[0].shape[1:])
    if len(model_dimensions) == 1:
        input_length = model_dimensions[0]
    elif len(model_dimensions) == 2 and 1 in model_dimensions:
        input_length = next(d for d in model_dimensions if d != 1)
    else:
        raise ValueError(
            f"unsupported model input shape: {model.inputs[0].shape}; "
            "expected (T,), (T,1), or (1,T)")
    if input_length is None:
        raise ValueError("model input length is dynamic; expected 875 samples")
    input_length = int(input_length)
    if input_length != 875:
        print(f"warning: model input length is {input_length}, not 875")

    signature = [tf.TensorSpec(input_shape, tf.float32, name="signal")]
    tf2onnx.convert.from_keras(
        model,
        input_signature=signature,
        opset=args.opset,
        output_path=str(output_path),
    )

    config_path = Path(args.config) if args.config else output_path.with_suffix(".json")
    config = {
        "input_length": input_length,
        "input_mean": args.input_mean,
        "input_std": args.input_std,
        "source_model": str(h5_path),
        "source_description": "Zenodo 5590603 resnet_ppg_nonmixed.h5",
        "input_sample_rate_hz": 125,
        "experimental_only": True,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"ONNX saved: {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Keras input: {model.input_shape}; output: {model.output_shape}")


if __name__ == "__main__":
    main()
