# Experimental Camera Blood Pressure

This server implements the deployment part of the following path:

`camera frames -> forehead ROI -> rPPG/BVP waveform -> quality gate -> rPPG-BP ONNX model -> SBP/DBP`

It does not create a BP value from heart rate. Without a configured, trained ONNX
model, the `blood_pressure` task returns waveform quality only.

## Offline validation first

Use a recorded MP4 before testing a camera stream:

```powershell
python server/bp_offline.py C:\data\subject_001.mp4 --window-seconds 7
```

The command writes `subject_001.rppg.npz`, containing `waveform`, `timestamps`,
`fs`, and `snr`. Compare it with synchronized reference PPG before connecting a
BP model.

## Model contract

Reproduce the official rPPG-BP preprocessing and train/test split first. Export
the trained model to ONNX with one input shaped `[B,T]` or `[B,1,T]`; the output
must contain `[SBP, DBP]` in mmHg. Start the service with:

```powershell
python server/main.py --bp-model C:\models\rppg_bp.onnx --bp-model-config C:\models\rppg_bp.json
```

Optional model config example:

```json
{
  "input_length": 900,
  "input_mean": 0.0,
  "input_std": 1.0
}
```

`input_length` and normalization must match training exactly. The model is not
considered valid until it is evaluated with a subject-independent split and
SBP/DBP MAE, RMSE, Pearson correlation, and Bland-Altman analysis.

## Use the pretrained Zenodo ResNet model

Download `resnet_ppg_nonmixed.h5` from [Zenodo record 5590603](https://zenodo.org/records/5590603),
then install the export dependencies and convert it:

```powershell
pip install tensorflow tf2onnx onnx onnxruntime
python server/convert_resnet_bp.py `
  --h5 C:\models\resnet_ppg_nonmixed.h5 `
  --output C:\models\resnet_ppg_nonmixed.onnx
```

The converter also writes `resnet_ppg_nonmixed.json`. Start the server with both
files:

```powershell
python server/main.py `
  --bp-model C:\models\resnet_ppg_nonmixed.onnx `
  --bp-model-config C:\models\resnet_ppg_nonmixed.json
```

The model expects 875 samples at 125 Hz. The deployment adapter resamples the
latest rPPG waveform to that length and accepts either one `[SBP, DBP]` output
or two separate output tensors. This is a compatibility/prototype path: the
Zenodo model was trained on PPG/rPPG data with preprocessing that may differ
from this project's POS signal.

## Desktop test

Start the server, then run:

```powershell
python server/desktop_client.py --fps 15
```

The client uses `capture_timestamp_ms`, continuously uploads frames, and queries
`blood_pressure` every two seconds. A measurement needs a stable face for roughly
7 seconds. Results are experimental only and must never be used for diagnosis,
treatment, or emergency decisions.
