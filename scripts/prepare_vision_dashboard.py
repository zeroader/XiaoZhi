"""Prepare the 320x240 vision dashboard background.

The supplied artwork is kept intact except for values/status captions that
must be rendered by LVGL at runtime.  OpenCV inpainting removes those fixed
characters without changing the surrounding illustrations.
"""
from pathlib import Path

import cv2


root = Path(__file__).resolve().parents[1]
source = root / "main" / "assets" / "vision_dashboard.png"
target = root / "main" / "assets" / "vision" / "vision_dashboard_base.png"
target.parent.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(source), cv2.IMREAD_COLOR)
if image is None:
    raise SystemExit(f"cannot read {source}")

image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
mask = image[:, :, 0] * 0


def clear(x1, y1, x2, y2):
    mask[y1:y2, x1:x2] = 255


# Dynamic value areas, mapped from the supplied 1448x1086 artwork to 320x240.
# Leave the labels/icons and the decorative graphs in the fixed background.
clear(27, 44, 57, 65)       # heart-rate number only; keep the printed unit
clear(12, 150, 67, 171)     # blood-pressure number only; keep printed mmHg
clear(248, 102, 312, 120)   # posture status caption
clear(248, 213, 312, 229)   # emotion status caption

# Inpaint with a small radius so card gradients and borders remain natural.
result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
if not cv2.imwrite(str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
    raise SystemExit(f"cannot write {target}")
print(target)
