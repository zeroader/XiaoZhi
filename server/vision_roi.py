"""Crop the dashboard ROI before inference; keep response boxes in camera coordinates."""

import copy


def crop_recognition_input(rgb, roi_text):
    if roi_text is None:
        return rgb, None
    try:
        values = [int(value) for value in roi_text.split(",")]
        if len(values) != 4:
            raise ValueError
        x, y, width, height = values
    except (TypeError, AttributeError, ValueError):
        raise ValueError("vision_roi must be x,y,width,height integers") from None
    image_height, image_width = rgb.shape[:2]
    if (x < 0 or y < 0 or width <= 0 or height <= 0 or
            x + width > image_width or y + height > image_height):
        raise ValueError("vision_roi is outside the uploaded image")
    return rgb[y:y + height, x:x + width].copy(), (x, y, width, height)


def camera_coordinates(result, roi):
    if roi is None:
        return result
    # Do not mutate cached face coordinates used by rPPG and subsequent tasks.
    result = copy.deepcopy(result)
    face = result.get("face")
    if face and face.get("bbox"):
        face["bbox"]["x"] += roi[0]
        face["bbox"]["y"] += roi[1]
    return result
